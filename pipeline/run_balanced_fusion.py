"""Run the development-selected, frozen balanced preset without ground truth.

python pipeline/run_balanced_fusion.py --sequence V1_01_easy \
  --cpu-python <numpy/scipy/Pillow/skimage/python> --gpu-python <torch/python>
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence',required=True,choices=['V1_01_easy','V1_02_medium','V1_03_difficult'])
    parser.add_argument('--cpu-python',default=sys.executable)
    parser.add_argument('--gpu-python',default=sys.executable)
    parser.add_argument('--work',help='New output root; does not replace legacy maps')
    parser.add_argument('--rebuild',action='store_true',help='Explicitly regenerate prepared cache when code/settings changed')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    preset=json.loads((root/'pipeline/fusion_balanced.json').read_text())
    script=str(root/'pipeline/fusion_v2.py')
    common=['--sequence',args.sequence]
    output=Path(args.work) if args.work else root/'outputs'/args.sequence/'stage8_fusion_balanced'
    common+=['--work',str(output)]
    prepare=[args.cpu_python,script,'prepare',*common,'--voxel',str(preset['voxel_m']),
             '--trunc',str(preset['trunc_m']),'--zmax',str(preset['zmax_m']),
             '--confidence-mode','none' if preset['variant'] in ['uniform','surfel_uniform'] else 'temporal']
    if args.rebuild: prepare+=['--rebuild']
    integrate=[args.gpu_python,script,'integrate',*common,'--variant',preset['variant']]
    extract=[args.cpu_python,script,'extract',*common,'--variant',preset['variant'],
             '--min-surface-support',str(preset['min_surface_support']),
             '--min-component-points',str(preset['min_component_points']),
             '--min-component-faces',str(preset['min_component_faces'])]
    for command in [prepare,integrate,extract]:
        print('Running',command[2],flush=True)
        subprocess.run(command,cwd=root,check=True)


if __name__=='__main__': main()
