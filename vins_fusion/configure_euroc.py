"""Use this sequence's ASL pinhole calibration, not a different rig's defaults."""
from __future__ import print_function
import os
import yaml


def configure(data_root, config_root):
    config_path = os.path.join(config_root, 'euroc_stereo_imu_config.yaml')
    with open(config_path) as handle:
        config = handle.read()
    for index in [0, 1]:
        name = 'cam' + str(index)
        with open(os.path.join(data_root, 'mav0', name, 'sensor.yaml')) as handle:
            sensor = yaml.safe_load(handle)
        if sensor['camera_model'] != 'pinhole' or sensor['distortion_model'] != 'radial-tangential':
            raise ValueError('Expected EuRoC pinhole/radial-tangential calibration')
        fx, fy, cx, cy = sensor['intrinsics']
        k1, k2, p1, p2 = sensor['distortion_coefficients']
        width, height = sensor['resolution']
        camera = ('%YAML:1.0\nmodel_type: PINHOLE\ncamera_name: ' + name +
                  '\nimage_width: %d\nimage_height: %d\n' % (width, height) +
                  'distortion_parameters: {k1: %.16g, k2: %.16g, p1: %.16g, p2: %.16g}\n' % (k1,k2,p1,p2) +
                  'projection_parameters: {fx: %.16g, fy: %.16g, cx: %.16g, cy: %.16g}\n' % (fx,fy,cx,cy))
        with open(os.path.join(config_root, name+'_mei.yaml'), 'w') as handle:
            handle.write(camera)
        start = config.index('body_T_' + name + ':')
        end = config.index(']', start) + 1
        matrix = 'body_T_' + name + ': !!opencv-matrix\n   rows: 4\n   cols: 4\n   dt: d\n   data: [' + ', '.join('%.16g' % v for v in sensor['T_BS']['data']) + ']'
        config = config[:start] + matrix + config[end:]
    with open(config_path, 'w') as handle:
        handle.write(config)


if __name__ == '__main__':
    configure('/data', '/tmp/euroc')
