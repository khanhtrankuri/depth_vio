"""Check the unit/frame contract between official VINS CSV and the mapper."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'vins_fusion'))
sys.path.insert(0, str(ROOT/'pipeline'))
from convert_trajectory import convert
from fusion_geometry import PoseStream
from run_vins_fusion import trajectory_summary


class VinsTrajectoryTests(unittest.TestCase):
    def test_short_partial_run_is_not_reported_as_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            camera, trajectory = Path(directory)/'images.csv', Path(directory)/'tum.txt'
            camera.write_text('#timestamp,filename\n1000000000,a.png\n10000000000,b.png\n')
            trajectory.write_text('1 0 0 0 0 0 0 1\n2 0 0 0 0 0 0 1\n')
            with self.assertRaises(ValueError): trajectory_summary(trajectory, camera)
            trajectory.write_text('1 0 0 0 0 0 0 1\n10 0 0 0 0 0 0 1\n')
            self.assertEqual(trajectory_summary(trajectory, camera)['last_image_lag_s'], 0)

    def test_ns_and_wxyz_are_converted_to_mapper_imu_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/'vio.csv', Path(directory)/'tum.txt'
            source.write_text('1400000000000000000,1,2,3,0.70710678,0,0,0.70710678,0,0,0,\n'
                              '1400000000100000000,3,2,3,0.70710678,0,0,0.70710678,0,0,0,\n')
            self.assertEqual(convert(source, target), 2)
            self.assertIn('1400000000.100000000', target.read_text())
            T, _ = PoseStream(target).at_ns(1400000000050000000)
            np.testing.assert_allclose(T[:3,3], [2,2,3], atol=1e-5)
            np.testing.assert_allclose(T[:3,:3] @ [1,0,0], [0,1,0], atol=1e-8)

    def test_restart_invalid_quaternion_and_empty_output_fail(self):
        for content in ['', '100,0,0,0,0,0,0,0,\n',
                        '100,0,0,0,1,0,0,0,\n100,1,0,0,1,0,0,0,\n',
                        'nan,0,0,0,1,0,0,0,\n']:
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                source, target = Path(directory)/'vio.csv', Path(directory)/'tum.txt'
                source.write_text(content)
                with self.assertRaises(ValueError): convert(source, target)
                self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
