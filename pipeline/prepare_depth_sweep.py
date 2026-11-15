"""Prepare immutable noise/frequency views of an existing IGEV depth export.

No model/GT input. Gaussian information multiplier w corresponds to sigma / sqrt(w)
before Huber loss. Frequencies select only exposures consumed by VINS's existing
every-other-frame backend; the backend's frequency itself is not changed.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import numpy as np


def select_exposures(index, camera_times, hz):
    stride={2.:1,1.:2,.5:4}[float(hz)]
    eligible=set(camera_times[1::2])
    available=[(t,name) for t,name in index if t in eligible]
    return available[::stride]


def scale_binary(source,target,sigma_multiplier):
    with open(source,'rb') as handle:
        header=handle.read(8);w,h=struct.unpack('<II',header)
        data=np.frombuffer(handle.read(),dtype='<f4').copy()
    if data.size!=w*h*2:raise ValueError('Invalid source depth binary')
    data.reshape(-1,2)[:,1]*=sigma_multiplier
    with open(target,'wb') as handle:handle.write(header);handle.write(data.tobytes())


def link(source,target):
    if target.exists():return
    try:os.link(source,target)
    except OSError:shutil.copyfile(source,target)


def prepare(source,camera_csv,output):
    source=Path(source).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    original=json.loads((source/'manifest.json').read_text())
    with open(camera_csv) as handle:
        camera_times=[int(row[0]) for row in csv.reader(line for line in handle if not line.startswith('#')) if row]
    index=[(int(t),name) for t,name in (line.split() for line in (source/'index.txt').read_text().splitlines())]
    eligible=select_exposures(index,camera_times,2.)
    fingerprint=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()
    configs=[]
    for weight in [.25,1.,4.]:
        scale=1/math.sqrt(weight);label=str(weight).replace('.','p')
        bank=output/('sigma_'+str(scale).replace('.','p'));bank.mkdir(exist_ok=True)
        for i,(t,name) in enumerate(eligible):
            target=bank/name
            if not target.exists():
                if scale==1:link(source/name,target)
                else:scale_binary(source/name,target,scale)
            if i%100==0:print('Noise view',weight,i,'/',len(eligible),flush=True)
        for hz in [.5,1.,2.]:
            identifier='w'+label+'_hz'+str(hz).replace('.','p')
            folder=output/identifier;folder.mkdir(exist_ok=True)
            selected=select_exposures(index,camera_times,hz)
            settings=dict(identifier=identifier,weight=weight,sigma_multiplier=scale,hz=hz,
                          frames=len(selected),phase=0,source_manifest_sha256=fingerprint)
            if (folder/'manifest.json').exists():
                existing=json.loads((folder/'manifest.json').read_text())
                if existing.get('sweep')!=settings:raise ValueError('Existing depth view configuration changed')
            for t,name in selected:link(bank/name,folder/name)
            manifest=dict(original,frames=len(selected),sigma_px=original['sigma_px']*scale,
                          relative_sigma=original['relative_sigma']*scale,sweep=settings,
                          nominal_weight_note='1/sigma_multiplier^2 before Huber; not an independent covariance claim')
            (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
            (folder/'index.txt').write_text('\n'.join(str(t)+' '+name for t,name in selected)+'\n')
            configs.append(dict(settings,depth_dir=str(folder)))
    (output/'configs.json').write_text(json.dumps(configs,indent=2))
    return configs


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--camera-csv',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();prepare(args.source,args.camera_csv,args.output)
