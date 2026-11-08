"""Frustum-bounded weighted projective TSDF; no GT or scene-specific world box."""
import time
import numpy as np
import torch
from fusion_geometry import frustum_bounds


class FrustumTSDF:
    def __init__(self, lo, hi, camera, voxel=.03, trunc=.12, device='cuda', chunk=1_000_000):
        if voxel<=0 or trunc<voxel:
            raise ValueError('Need 0 < voxel <= trunc')
        self.camera=camera; self.voxel=float(voxel); self.trunc=float(trunc)
        self.lo=np.asarray(lo,dtype=np.float64)
        self.dims=np.ceil((np.asarray(hi,dtype=np.float64)-self.lo)/voxel-1e-9).astype(np.int64)
        if (self.dims<=0).any(): raise ValueError('Empty volume')
        self.device=torch.device(device); self.chunk=chunk
        shape=tuple(self.dims)
        self.tsdf=torch.ones(shape,device=self.device)
        self.weight=torch.zeros(shape,device=self.device)
        self.count=torch.zeros(shape,dtype=torch.int16,device=self.device)
        self.surface_count=torch.zeros(shape,dtype=torch.int16,device=self.device)
        self.color=torch.zeros(shape,device=self.device)
        self.origin=torch.tensor(self.lo,dtype=torch.float32,device=self.device)

    def integrate(self, depth, weights, gray, T_WC):
        cam=self.camera
        if depth.shape!=(cam.height,cam.width) or weights.shape!=depth.shape:
            raise ValueError('Depth/weight shape does not match camera calibration')
        a,b=frustum_bounds(T_WC,cam,.2,float(depth.max())+self.trunc)
        start=np.maximum(np.floor((a-self.lo)/self.voxel).astype(int)-1,0)
        end=np.minimum(np.ceil((b-self.lo)/self.voxel).astype(int)+1,self.dims)
        size=end-start
        if (size<=0).any(): return dict(projected_voxels=0,updated_voxels=0)
        D=torch.as_tensor(depth,dtype=torch.float32,device=self.device)
        I=torch.as_tensor(weights,dtype=torch.float32,device=self.device)
        C=torch.as_tensor(gray,dtype=torch.float32,device=self.device)/255.
        # Row-vector p_C = (p_W - t_WC) R_WC.
        R=torch.tensor(T_WC[:3,:3],dtype=torch.float32,device=self.device)
        t=torch.tensor(T_WC[:3,3],dtype=torch.float32,device=self.device)
        F=self.tsdf.view(-1); W=self.weight.view(-1); N=self.count.view(-1)
        S=self.surface_count.view(-1); G=self.color.view(-1)
        n=int(size.prod()); updates=torch.zeros((),device=self.device,dtype=torch.int64)
        for begin in range(0,n,self.chunk):
            j=torch.arange(begin,min(begin+self.chunk,n),device=self.device)
            x=j//int(size[1]*size[2])+int(start[0])
            y=(j//int(size[2]))%int(size[1])+int(start[1])
            z=j%int(size[2])+int(start[2])
            idx=(x*int(self.dims[1])+y)*int(self.dims[2])+z
            p=(torch.stack((x,y,z),dim=1).float()+.5)*self.voxel+self.origin
            pc=(p-t)@R; zz=pc[:,2]
            u=torch.round(cam.fx*pc[:,0]/zz.clamp(min=1e-6)+cam.cx).long()
            v=torch.round(cam.fy*pc[:,1]/zz.clamp(min=1e-6)+cam.cy).long()
            visible=(zz>.2)&(u>=0)&(u<cam.width)&(v>=0)&(v<cam.height)
            k=torch.where(visible)[0]
            ui=u[k]; vi=v[k]; measured=D[vi,ui]; confidence=I[vi,ui]
            sdf=measured-zz[k]
            accept=torch.isfinite(measured)&torch.isfinite(confidence)&(measured>0)&(confidence>0)&(sdf > -self.trunc)
            k=k[accept]; indices=idx[k]; wi=confidence[accept]
            residual=sdf[accept]
            total=W[indices]+wi
            F[indices]=(F[indices]*W[indices]+(residual/self.trunc).clamp(-1,1)*wi)/total
            G[indices]=(G[indices]*W[indices]+C[v[k],u[k]]*wi)/total
            W[indices]=total
            # One count per actual input frame, regardless of proxy confidence.
            N[indices]=(N[indices].to(torch.int32)+1).clamp(max=32767).to(torch.int16)
            near=indices[residual.abs()<2*self.voxel]
            S[near]=(S[near].to(torch.int32)+1).clamp(max=32767).to(torch.int16)
            updates+=len(indices)
        return dict(projected_voxels=n,updated_voxels=int(updates.item()))

    def save(self,path):
        # np.savez (uncompressed) keeps export fast and reproducible.
        np.savez(path,tsdf=self.tsdf.cpu().numpy(),weight=self.weight.cpu().numpy(),
                 count=self.count.cpu().numpy(),surface_count=self.surface_count.cpu().numpy(),
                 color=self.color.cpu().numpy(),origin=self.lo,voxel=self.voxel)


class VoxelSurfels:
    """Fast point-surface backend; one weighted vote per voxel per input frame.

    Keeps observed surface centroids without synthesizing TSDF zero crossings.
    This exports a point surface, not a watertight mesh. Counts are temporal
    support, not the number of pixels hitting a voxel in a single image.
    """
    def __init__(self,lo,hi,camera,voxel=.03,trunc=.12,device='cuda',chunk=1_000_000):
        self.camera=camera; self.voxel=voxel; self.lo=np.asarray(lo,dtype=np.float64)
        self.dims=np.ceil((np.asarray(hi)-self.lo)/voxel-1e-9).astype(np.int64)
        self.device=torch.device(device); n=int(self.dims.prod())
        self.xyz=torch.zeros((n,3),device=self.device)
        self.weight=torch.zeros(n,device=self.device)
        self.count=torch.zeros(n,device=self.device,dtype=torch.int16)
        self.color=torch.zeros(n,device=self.device)
        self.rays=torch.tensor(camera.rays(),device=self.device)
        self.origin=torch.tensor(self.lo,dtype=torch.float32,device=self.device)

    def integrate(self,depth,weights,gray,T_WC):
        valid=(depth>0)&np.isfinite(depth)&np.isfinite(weights)&(weights>0)
        mask=torch.as_tensor(valid,device=self.device)
        d=torch.as_tensor(depth,device=self.device)[mask]
        w=torch.as_tensor(weights,device=self.device)[mask]
        g=torch.as_tensor(gray,dtype=torch.float32,device=self.device)[mask]/255.
        p=self.rays[mask]*d[:,None]
        R=torch.tensor(T_WC[:3,:3],dtype=torch.float32,device=self.device)
        t=torch.tensor(T_WC[:3,3],dtype=torch.float32,device=self.device)
        p=p@R.T+t
        ijk=torch.floor((p-self.origin)/self.voxel).long()
        dims=torch.tensor(self.dims,device=self.device)
        good=((ijk>=0)&(ijk<dims)).all(dim=1)
        p=p[good]; w=w[good]; g=g[good]; ijk=ijk[good]
        if not len(p): return dict(projected_voxels=0,updated_voxels=0)
        ids=(ijk[:,0]*int(self.dims[1])+ijk[:,1])*int(self.dims[2])+ijk[:,2]
        unique,inverse,pixel_count=torch.unique(ids,return_inverse=True,return_counts=True)
        sw=torch.zeros(len(unique),device=self.device); sw.scatter_add_(0,inverse,w)
        # Normalize all within-frame pixels into one centroid and one information
        # contribution. Image resolution cannot masquerade as repeated views.
        frame_w=sw/pixel_count
        for axis in range(3):
            total=torch.zeros(len(unique),device=self.device)
            total.scatter_add_(0,inverse,w*p[:,axis])
            self.xyz[unique,axis]+=total/sw*frame_w
        total_g=torch.zeros(len(unique),device=self.device); total_g.scatter_add_(0,inverse,w*g)
        self.color[unique]+=total_g/sw*frame_w
        self.weight[unique]+=frame_w
        self.count[unique]=(self.count[unique].to(torch.int32)+1).clamp(max=32767).to(torch.int16)
        return dict(projected_voxels=len(p),updated_voxels=len(unique))

    def save(self,path):
        good=self.count>0
        weight=self.weight[good]
        np.savez(path,points=(self.xyz[good]/weight[:,None]).cpu().numpy(),
                 color=(self.color[good]/weight).cpu().numpy(),
                 count=self.count[good].cpu().numpy(),weight=weight.cpu().numpy(),
                 origin=self.lo,voxel=self.voxel,representation='voxel surfel centroids, not mesh')
