"""Export cached rectified IGEV++ disparity as raw-cam0 inverse-depth observations.

No GT input. Binary: uint32 width,height then float32 [inverse_depth, sigma] per pixel.
Sigma is a conservative proxy, not calibrated uncertainty. Cached exposures only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import cv2
import numpy as np
from common import ROOT, stereo_rectify, cam_calib


def raw_inverse_map(disparity,valid,rect,K,D,zmin=.2,zmax=4.,sigma_px=2.,relative_sigma=.05):
    h,w=disparity.shape
    v,u=np.indices((h,w),dtype=np.float32)
    raw_pixels=np.stack([u,v],-1).reshape(-1,1,2)
    rays=cv2.undistortPoints(raw_pixels,K,D).reshape(h,w,2)
    rays=np.concatenate([rays,np.ones((h,w,1))],axis=2)
    rect_rays=rays@rect['R0'].T
    scale=rect_rays[...,2]
    mapx=(rect['fx']*rect_rays[...,0]/scale+rect['cx']).astype(np.float32)
    mapy=(rect['fy']*rect_rays[...,1]/scale+rect['cy']).astype(np.float32)
    disparity=np.asarray(disparity,np.float32)
    d=cv2.remap(disparity,mapx,mapy,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    mask=cv2.remap(np.asarray(valid,np.uint8),mapx,mapy,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)>0
    inverse=scale*d/(rect['fx']*rect['baseline'])
    sigma=np.maximum(relative_sigma*inverse,np.abs(scale)*sigma_px/(rect['fx']*rect['baseline']))
    good=mask&np.isfinite(inverse)&np.isfinite(sigma)&(inverse>=1/zmax)&(inverse<=1/zmin)&(scale>0)
    result=np.zeros((h,w,2),dtype='<f4')
    result[...,0][good]=inverse[good];result[...,1][good]=sigma[good]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--sigma-px',type=float,default=2.)
    parser.add_argument('--relative-sigma',type=float,default=.05)
    args=parser.parse_args()
    if args.sigma_px<=0 or args.relative_sigma<=0: parser.error('Noise proxies must be positive')
    from common import SEQ
    cache=Path(ROOT)/'outputs'/SEQ/'stage8_fusion/disp'
    files=sorted(cache.glob('*.npz'))
    if not files: raise FileNotFoundError('No IGEV++ disparity cache: '+str(cache))
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if (output/'index.txt').exists(): raise FileExistsError('Use a fresh depth export directory')
    rect=stereo_rectify();K,D,_,_=cam_calib('cam0')
    sources={};index=[];counts=[]
    for i,path in enumerate(files):
        with np.load(path) as a:
            if a['disp'].shape!=rect['map0'][0].shape or a['valid'].shape!=a['disp'].shape:
                raise ValueError('Disparity/mask dimensions do not match camera calibration: '+str(path))
            observations=raw_inverse_map(a['disp'],a['valid'],rect,K,D,sigma_px=args.sigma_px,relative_sigma=args.relative_sigma)
        h,w=observations.shape[:2];target=output/(path.stem+'.bin')
        with target.open('wb') as handle:
            handle.write(struct.pack('<II',w,h));handle.write(observations.tobytes())
        sources[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        index.append(path.stem+' '+target.name);counts.append(int((observations[...,0]>0).sum()))
        if i%100==0: print('depth export',i,'/',len(files),flush=True)
    manifest=dict(sequence=SEQ,frames=len(files),sigma_px=args.sigma_px,relative_sigma=args.relative_sigma,
                  min_depth_m=.2,max_depth_m=4.,coordinate_frame='raw cam0 optical Z; inverse metres',
                  sampling='nearest rectified pixel; C++ accepts smooth valid 3x3 raw neighbourhood',
                  uncertainty='proxy only, not calibrated; not independent of stereo',
                  source_sha256=sources,valid_pixels_mean=float(np.mean(counts)),
                  calibration_sha256={c:hashlib.sha256((Path(ROOT)/'data'/SEQ/'mav0'/c/'sensor.yaml').read_bytes()).hexdigest() for c in ['cam0','cam1']})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (output/'index.txt').write_text('\n'.join(index)+'\n')
    print('Exported',len(files),'depth exposures to',output)


if __name__=='__main__': main()
