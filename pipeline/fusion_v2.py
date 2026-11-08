"""Balanced depth/VIO mapping with independently runnable preparation/GPU/mesh phases.

Examples from repository root:
 python pipeline/fusion_v2.py prepare --sequence V1_01_easy
 python pipeline/fusion_v2.py integrate --sequence V1_01_easy --variant temporal
 python pipeline/fusion_v2.py extract --sequence V1_01_easy --variant temporal

No Leica, GT trajectory, GT-derived rotation correction, or scan registration is
read by this module. Existing depth inference caches and raw OpenVINS poses are
inputs. Separate phases also support machines with split CPU/GPU dependencies.
"""
import argparse
from collections import deque
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from fusion_geometry import Camera, PoseStream, calibration, depth_from_disparity, transform
from fusion_confidence import ConfidenceConfig, spatial_weights, temporal_weights

ROOT=Path(__file__).resolve().parents[1]
SEQUENCES=['V1_01_easy','V1_02_medium','V1_03_difficult']


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''): h.update(block)
    return h.hexdigest()


def write_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def validate_confidence_variant(variant,mode):
    if variant in ['temporal','surfel_temporal'] and mode!='temporal':
        raise ValueError('Temporal variant requires a cache prepared with --confidence-mode temporal')
    if variant=='spatial' and mode=='none':
        raise ValueError('Spatial variant requires spatial/temporal confidence preparation')


def workdir(args):
    return Path(args.work) if args.work else ROOT/'outputs'/args.sequence/'stage8_fusion_v2'


def rectification_coordinates(cam,R0,sensor):
    # Independent pinhole+radtan inverse rectification, color only (not inference).
    rays=cam.rays()@R0
    x=rays[...,0]/rays[...,2]; y=rays[...,1]/rays[...,2]
    k1,k2,p1,p2=sensor['distortion_coefficients']
    r=x*x+y*y; radial=1+k1*r+k2*r*r
    xx=x*radial+2*p1*x*y+p2*(r+2*x*x)
    yy=y*radial+p1*(r+2*y*y)+2*p2*x*y
    fx,fy,cx,cy=sensor['intrinsics']
    return np.stack([fy*yy+cy,fx*xx+cx])


def rectified_gray(image_path,cam,R0,sensor,coordinates=None):
    from PIL import Image
    from scipy.ndimage import map_coordinates
    raw=np.asarray(Image.open(image_path).convert('L'))
    if coordinates is None: coordinates=rectification_coordinates(cam,R0,sensor)
    return map_coordinates(raw,coordinates,order=1,mode='constant',cval=0).astype(np.uint8)


def prepare(args):
    work=workdir(args); target=work/'prepared'; target.mkdir(parents=True,exist_ok=True)
    pose_path=Path(args.poses) if args.poses else ROOT/'outputs'/args.sequence/'stage7_vio/openvins_traj.txt'
    stream=PoseStream(pose_path)
    cam,ext,R0,sensor=calibration(ROOT,args.sequence); rays=cam.rays()
    color_coordinates=rectification_coordinates(cam,R0,sensor)
    cfg=ConfidenceConfig(temporal_neighbors=args.neighbors)
    cache=sorted((ROOT/'outputs'/args.sequence/'stage8_fusion/disp').glob('*.npz'))
    config=dict(sequence=args.sequence,camera=asdict(cam),confidence=asdict(cfg),confidence_mode=args.confidence_mode,zmin=args.zmin,zmax=args.zmax,
                voxel=args.voxel,trunc=args.trunc,poses=str(pose_path.relative_to(ROOT)) if pose_path.is_relative_to(ROOT) else str(pose_path),
                poses_sha256=sha(pose_path),T_I_Crect=ext.tolist(),pose_interpolation='linear+SLERP; no extrapolation',
                coordinate_frame='native OpenVINS global; no GT-derived registration',
                covariance='blocks available but NOT treated as calibrated uncertainty or fused',
                timestamp_note='raw pose timestamps assumed IMU-time; online camera-IMU offset not exported by legacy run',
                legacy_cache_note='cache has no historical model/calibration fingerprint; provenance remains conditional',
                source_fingerprint={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in cache},
                code_fingerprint={p.name:sha(p) for p in [Path(__file__),ROOT/'pipeline/fusion_geometry.py',ROOT/'pipeline/fusion_confidence.py']})
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    manifest=target/'manifest.json'
    if manifest.exists() and not args.rebuild:
        previous=json.loads(manifest.read_text())
        if previous['fingerprint']==fingerprint:
            files=[target/(str(t)+'.npz') for t in previous['timestamps_ns']]
            if all(p.exists() for p in files):
                print('Prepared cache verified/reused:',len(files),flush=True); return
        raise RuntimeError('Prepared cache/config changed. Use a separate --work or explicit --rebuild.')
    past=deque(maxlen=cfg.temporal_neighbors)
    poses=[]; timestamps=[]; timings=[]; checks=[]; skipped=[]
    lo=np.full(3,np.inf); hi=-lo
    start=time.perf_counter()
    for i,path in enumerate(cache):
        t=int(path.stem); pose=stream.at_ns(t)
        if pose is None:
            skipped.append(t); continue
        frame_start=time.perf_counter(); T_WI,sync=pose; T_WC=T_WI@ext
        with np.load(path) as data:
            z=depth_from_disparity(data['disp'],data['valid'],cam,args.zmin,args.zmax)
        if args.confidence_mode=='none':
            sw=(z>0).astype(np.float32); tw=sw
            check=dict(agree=0,conflict=0,occluded=0,checked=0,rejected=0)
        else:
            sw,edge=spatial_weights(z,cam,cfg)
            if args.confidence_mode=='temporal':
                tw,check=temporal_weights(z,T_WC,list(past),cam,cfg,rays)
            else:
                tw=(z>0).astype(np.float32)
                check=dict(agree=0,conflict=0,occluded=0,checked=0,rejected=0)
        image=ROOT/'data'/args.sequence/'mav0/cam0/data'/(str(t)+'.png')
        gray=rectified_gray(image,cam,R0,sensor,color_coordinates)
        np.savez_compressed(target/(str(t)+'.npz'),depth=z,spatial=sw.astype(np.float16),
                            temporal=tw.astype(np.float16),gray=gray)
        # Native volume bounds depend only on input measured surfaces, never GT/scan.
        p=(rays*z[...,None])[::8,::8]; good=z[::8,::8]>0
        if good.any():
            p=transform(p[good],T_WC); lo=np.minimum(lo,p.min(0)); hi=np.maximum(hi,p.max(0))
        if args.confidence_mode=='temporal': past.append((z,T_WC,edge))
        poses.append(T_WC); timestamps.append(t)
        checks.append(dict(t_ns=t,**check,**sync))
        timings.append((time.perf_counter()-frame_start)*1000)
        if i%50==0: print('prepare',i,'/',len(cache),'ms/frame',round(np.mean(timings),1),flush=True)
    if not poses: raise RuntimeError('No depth/pose matches')
    lo=np.floor((lo-.2)/args.voxel)*args.voxel
    hi=np.ceil((hi+.2)/args.voxel)*args.voxel
    np.save(target/'poses_native.npy',np.asarray(poses)); np.save(target/'timestamps_ns.npy',np.asarray(timestamps,dtype=np.int64))
    write_json(manifest,dict(fingerprint=fingerprint,config=config,timestamps_ns=timestamps,skipped_timestamps_ns=skipped,
                             volume_lo=lo.tolist(),volume_hi=hi.tolist(),frames=checks,
                             prepare_ms_mean=float(np.mean(timings)),prepare_ms_p90=float(np.quantile(timings,.9)),
                             wall_s=time.perf_counter()-start))
    print('Prepared',len(poses),'frames, native bounds',lo,hi,flush=True)


def integrate(args):
    import torch
    from fusion_tsdf import FrustumTSDF, VoxelSurfels
    work=workdir(args); source=work/'prepared'
    manifest=json.loads((source/'manifest.json').read_text()); cfg=manifest['config']
    validate_confidence_variant(args.variant,cfg.get('confidence_mode','temporal'))
    cam=Camera(**cfg['camera']); poses=np.load(source/'poses_native.npy')
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    if args.device=='cuda': torch.cuda.reset_peak_memory_stats()
    out=work/args.variant; out.mkdir(parents=True,exist_ok=True)
    backend=VoxelSurfels if args.variant.startswith('surfel') else FrustumTSDF
    volume=backend(manifest['volume_lo'],manifest['volume_hi'],cam,cfg['voxel'],cfg['trunc'],args.device,args.chunk)
    times=[]; statistics=[]; start=time.perf_counter()
    for i,(t,T) in enumerate(zip(manifest['timestamps_ns'],poses)):
        with np.load(source/(str(t)+'.npz')) as a:
            z=a['depth']; gray=a['gray']
            weights=(z>0).astype(np.float32)
            if args.variant in ['spatial','temporal','surfel_temporal']: weights*=a['spatial'].astype(np.float32)
            if args.variant in ['temporal','surfel_temporal']: weights*=a['temporal'].astype(np.float32)
        if args.device=='cuda': torch.cuda.synchronize()
        tick=time.perf_counter()
        stats=volume.integrate(z,weights,gray,T)
        if args.device=='cuda': torch.cuda.synchronize()
        times.append((time.perf_counter()-tick)*1000); statistics.append(stats)
        if i%50==0: print('integrate',args.variant,i,'/',len(poses),'ms',round(np.mean(times),1),flush=True)
    volume.save(out/'volume.npz')
    write_json(out/'integration.json',dict(sequence=args.sequence,variant=args.variant,frames=len(poses),
        prepared_fingerprint=manifest['fingerprint'],device=str(volume.device),voxel_m=cfg['voxel'],trunc_m=cfg['trunc'],
        dims=volume.dims.tolist(),integrate_ms_mean=float(np.mean(times)),integrate_ms_p90=float(np.quantile(times,.9)),
        wall_s=time.perf_counter()-start,voxel_candidates_mean=float(np.mean([x['projected_voxels'] for x in statistics])),
        updated_voxels_mean=float(np.mean([x['updated_voxels'] for x in statistics])),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None,
        integration_code_sha256={p.name:sha(p) for p in [Path(__file__),ROOT/'pipeline/fusion_tsdf.py']},
        provenance='native VIO poses + cached stereo; no GT or scan read during integration'))
    print('Wrote',out/'volume.npz',flush=True)


def write_ply(path,vertices,faces,gray):
    vertex=np.empty(len(vertices),dtype=[('x','<f8'),('y','<f8'),('z','<f8'),('red','u1'),('green','u1'),('blue','u1')])
    for i,k in enumerate('xyz'): vertex[k]=vertices[:,i]
    g=np.uint8(np.clip(gray*255,0,255))
    for k in ['red','green','blue']: vertex[k]=g
    with open(path,'wb') as f:
        header='ply\nformat binary_little_endian 1.0\nelement vertex '+str(len(vertices))+'\nproperty double x\nproperty double y\nproperty double z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n'
        if faces is not None:
            header+='element face '+str(len(faces))+'\nproperty list uchar int vertex_indices\n'
        f.write((header+'end_header\n').encode()); vertex.tofile(f)
        if faces is not None:
            a=np.empty(len(faces),dtype=[('n','u1'),('idx','<i4',(3,))]); a['n']=3; a['idx']=faces; a.tofile(f)


def extract(args):
    if args.variant.startswith('surfel'):
        from scipy.spatial import cKDTree
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        out=workdir(args)/args.variant
        with np.load(out/'volume.npz') as a:
            mask=a['count']>=max(args.min_observations,args.min_surface_support)
            points=a['points'][mask]; gray=a['color'][mask]
        if args.min_component_points>0 and len(points):
            radius=2.2*float(json.loads((out/'integration.json').read_text())['voxel_m'])
            pairs=cKDTree(points).query_pairs(radius,output_type='ndarray')
            graph=coo_matrix((np.ones(len(pairs),np.uint8),(pairs[:,0],pairs[:,1])),shape=(len(points),len(points))).tocsr()
            _,labels=connected_components(graph,directed=False)
            sizes=np.bincount(labels); keep=sizes[labels]>=args.min_component_points
            points=points[keep]; gray=gray[keep]
        suffix=f'_cc{args.min_component_points}' if args.min_component_points else ''
        stem=f'surface_support{args.min_surface_support}'+suffix
        write_ply(out/(stem+'_cloud.ply'),points,None,gray)
        write_json(out/(stem+'.json'),dict(vertices=len(points),faces=0,representation='surfel point surface; no mesh',
                   min_observations=max(args.min_observations,args.min_surface_support),
                   min_surface_support=args.min_surface_support,min_component_points=args.min_component_points))
        print(stem,len(points),'surfels',flush=True)
        return
    from skimage.measure import marching_cubes
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    out=workdir(args)/args.variant
    start=time.perf_counter()
    with np.load(out/'volume.npz') as a:
        vol=a['tsdf']; count=a['count']; support=a['surface_count']; color=a['color']; origin=a['origin']; voxel=float(a['voxel'])
    mask=count>=args.min_observations
    if not mask.any() or vol[mask].min()>0: raise RuntimeError('No observed TSDF zero crossing')
    vertices,faces,_,_=marching_cubes(vol,level=0,mask=mask,allow_degenerate=False)
    lower=np.floor(vertices).astype(int); upper=np.minimum(np.ceil(vertices).astype(int),np.array(vol.shape)-1)
    a=tuple(lower.T); b=tuple(upper.T)
    # A marching-cubes vertex lies on a voxel edge: both edge endpoints must be observed.
    keep=(count[a]>=args.min_observations)&(count[b]>=args.min_observations)
    if args.min_surface_support>0:
        keep&=np.maximum(support[a],support[b])>=args.min_surface_support
    faces=faces[keep[faces].all(1)]
    removed_before_components=len(vertices)-int(keep.sum())
    if len(faces) and args.min_component_faces:
        edges=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
        graph=coo_matrix((np.ones(len(edges),np.uint8),(edges[:,0],edges[:,1])),shape=(len(vertices),len(vertices))).tocsr()
        _,labels=connected_components(graph,directed=False)
        face_labels=labels[faces[:,0]]
        sizes=np.bincount(face_labels,minlength=len(vertices))
        faces=faces[sizes[face_labels]>=args.min_component_faces]
    used=np.unique(faces); remap=np.full(len(vertices),-1,dtype=np.int64); remap[used]=np.arange(len(used))
    faces=remap[faces]; vertices=vertices[used]
    idx=np.minimum(np.rint(vertices).astype(int),np.array(vol.shape)-1)
    gray=color[tuple(idx.T)]
    vertices=(vertices+.5)*voxel+origin
    stem=f'mesh_support{args.min_surface_support}'
    write_ply(out/(stem+'.ply'),vertices,faces[:,::-1].copy(),gray)
    write_ply(out/(stem+'_cloud.ply'),vertices,None,gray)
    write_json(out/(stem+'.json'),dict(vertices=len(vertices),faces=len(faces),
        min_observations=args.min_observations,min_surface_support=args.min_surface_support,
        min_component_faces=args.min_component_faces,unobserved_or_unsupported_vertices_removed=removed_before_components,
        extraction_wall_s=time.perf_counter()-start))
    print(stem,len(vertices),'vertices',len(faces),'faces',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','integrate','extract'])
    parser.add_argument('--sequence',choices=SEQUENCES,required=True)
    parser.add_argument('--work',help='Isolated output root; default outputs/SEQ/stage8_fusion_v2')
    parser.add_argument('--poses',help='Raw TUM/OpenVINS export; defaults to raw pose+covariance file')
    parser.add_argument('--variant',choices=['uniform','spatial','temporal','surfel_uniform','surfel_temporal'],default='temporal')
    parser.add_argument('--zmin',type=float,default=.2); parser.add_argument('--zmax',type=float,default=4.)
    parser.add_argument('--voxel',type=float,default=.03); parser.add_argument('--trunc',type=float,default=.12)
    parser.add_argument('--neighbors',type=int,default=2); parser.add_argument('--rebuild',action='store_true')
    parser.add_argument('--confidence-mode',choices=['none','spatial','temporal'],default='temporal',help='Preparation: skip unused confidence work for uniform backends')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda'); parser.add_argument('--chunk',type=int,default=1_000_000)
    parser.add_argument('--min-observations',type=int,default=2); parser.add_argument('--min-surface-support',type=int,default=1)
    parser.add_argument('--min-component-faces',type=int,default=30)
    parser.add_argument('--min-component-points',type=int,default=0,help='Surfel only: remove small spatial components; 0 disables')
    args=parser.parse_args()
    if args.neighbors<1: parser.error('--neighbors must be positive')
    if args.zmin<=0 or args.zmax<=args.zmin: parser.error('Need 0 < zmin < zmax')
    globals()[args.phase](args)


if __name__=='__main__': main()
