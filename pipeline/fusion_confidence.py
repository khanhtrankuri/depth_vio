"""Conservative confidence proxies; these weights are NOT calibrated uncertainty."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter


@dataclass(frozen=True)
class ConfidenceConfig:
    disparity_sigma_px: float = .5
    weight_reference_sigma_m: float = .05
    weight_min: float = .15
    weight_max: float = 3.
    temporal_absolute_m: float = .06
    temporal_relative: float = .025
    temporal_neighbors: int = 2
    temporal_conflict_weight: float = .2
    edge_relative: float = .08
    edge_weight: float = .3


def depth_edges(z, relative=.08):
    good = z>0
    near = minimum_filter(np.where(good,z,np.inf),size=3,mode='nearest')
    far = maximum_filter(z,size=3,mode='nearest')
    return good&((far-near)>relative*np.maximum(z,.2))


def spatial_weights(z, cam, cfg):
    sigma = cfg.disparity_sigma_px*z*z/(cam.fx*cam.baseline)
    weights = np.clip((cfg.weight_reference_sigma_m/np.maximum(sigma,.005))**2,
                      cfg.weight_min,cfg.weight_max)
    edges = depth_edges(z,cfg.edge_relative)
    weights[edges] *= cfg.edge_weight
    weights[z<=0] = 0
    return weights.astype(np.float32),edges


def temporal_weights(z, T_WC, history, camera, cfg, rays=None):
    """Backward-project current surface into past depth maps.

    If a nearer past surface occludes the current point, that is NO evidence
    against it. Missing/out-of-view/edge samples are also neutral. A current point
    in front of a valid past surface is contradicted only away from depth edges.
    Never average warped depths or count reused depth as independent evidence.
    """
    if rays is None: rays=camera.rays()
    v,u=np.nonzero(z>0)
    weights=np.ones_like(z,dtype=np.float32)
    if len(v)==0 or not history:
        weights[z<=0]=0
        return weights,dict(agree=0,conflict=0,occluded=0,checked=0,rejected=0)
    current=rays[v,u]*z[v,u,None]
    agreement=np.zeros(len(v),np.uint8); conflict=np.zeros(len(v),np.uint8)
    total_occ=total_check=0
    for item in history[-cfg.temporal_neighbors:]:
        zn,Tn,edge=item
        T=np.linalg.inv(Tn)@T_WC
        p=current@T[:3,:3].T+T[:3,3]
        positive=p[:,2]>.2
        uu=np.rint(camera.fx*p[:,0]/np.maximum(p[:,2],1e-6)+camera.cx).astype(np.int64)
        vv=np.rint(camera.fy*p[:,1]/np.maximum(p[:,2],1e-6)+camera.cy).astype(np.int64)
        keep=positive&(uu>=0)&(uu<camera.width)&(vv>=0)&(vv<camera.height)
        ids=np.flatnonzero(keep); uu=uu[keep]; vv=vv[keep]
        observed=zn[vv,uu]; expected=p[ids,2]
        check=(observed>0)&(~edge[vv,uu])
        tolerance=cfg.temporal_absolute_m+cfg.temporal_relative*np.minimum(observed,expected)
        residual=expected-observed
        same=check&(np.abs(residual)<=tolerance)
        # positive residual -> current point hidden behind old foreground: neutral.
        occ=check&(residual>tolerance)
        bad=check&(residual < -tolerance)
        agreement[ids[same]]+=1; conflict[ids[bad]]+=1
        total_occ+=int(occ.sum()); total_check+=int(check.sum())
    factor=(1.+.25*agreement)/(1.+(1./cfg.temporal_conflict_weight-1)*conflict)
    # Require two independent past frames and no agreement for hard rejection.
    reject=(conflict>=2)&(agreement==0)
    factor[reject]=0
    weights[v,u]=factor; weights[z<=0]=0
    return weights,dict(agree=int((agreement>0).sum()),conflict=int((conflict>0).sum()),
                        occluded=total_occ,checked=total_check,rejected=int(reject.sum()))
