import json
import pathlib
import subprocess

r = pathlib.Path('/catkin_ws/src/open_vins')
files = ['config/euroc_mav/estimator_config.yaml', 'config/euroc_mav/kalibr_imucam_chain.yaml',
         'config/euroc_mav/kalibr_imu_chain.yaml']
print(json.dumps(dict(commit=subprocess.check_output(['git', '-C', str(r), 'rev-parse', 'HEAD']).decode().strip(),
                      describe=subprocess.check_output(['git', '-C', str(r), 'describe', '--tags', '--always']).decode().strip(),
                      configs={f: (r/f).read_text() for f in files}), indent=2))
