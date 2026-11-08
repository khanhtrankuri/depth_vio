"""Physical controls for synchronization, occlusion and weighted fusion."""
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial.transform import Rotation
from fusion_geometry import Camera, PoseStream, depth_from_disparity
from fusion_confidence import ConfidenceConfig, depth_edges, temporal_weights


class GeometryTests(unittest.TestCase):
    def test_cannot_silently_run_temporal_variant_with_fast_uniform_cache(self):
        from fusion_v2 import validate_confidence_variant
        validate_confidence_variant('surfel_uniform','none')
        with self.assertRaises(ValueError): validate_confidence_variant('temporal','none')
        with self.assertRaises(ValueError): validate_confidence_variant('spatial','none')

    def test_pose_interpolates_translation_and_rotation_without_extrapolation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'poses.txt'
            q=Rotation.from_rotvec([[0,0,0],[0,0,np.pi/2]]).as_quat()
            rows=np.column_stack(([1000.,1002.],[[0.,0,0],[2.,0,0]],q))
            np.savetxt(path,rows,fmt='%.12f'); stream=PoseStream(path)
            T,info=stream.at_ns(1001_000_000_000,max_gap_s=3)
            np.testing.assert_allclose(T[:3,3],[1,0,0])
            np.testing.assert_allclose(T[:3,:3],Rotation.from_euler('z',45,degrees=True).as_matrix(),atol=1e-10)
            self.assertIsNone(stream.at_ns(999_000_000_000,max_gap_s=3))
            self.assertIsNone(stream.at_ns(1001_000_000_000,max_gap_s=.1))

    def test_invalid_disparity_is_never_integrated(self):
        cam=Camera(5,1,10,10,2,0,.1)
        d=np.array([[1,np.nan,np.inf,-1,0]],np.float32)
        z=depth_from_disparity(d,np.ones_like(d,bool),cam)
        np.testing.assert_array_equal(z,[[1,0,0,0,0]])

    def test_repeated_timestamp_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'poses.txt'
            np.savetxt(path,[[1,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,1]])
            with self.assertRaises(ValueError): PoseStream(path)


class VisibilityTests(unittest.TestCase):
    def setUp(self):
        self.cam=Camera(9,9,10,10,4,4,.1); self.cfg=ConfidenceConfig()
        self.T=np.eye(4); self.clear=np.zeros((9,9),bool)

    def test_consistent_plane_gets_support(self):
        z=np.full((9,9),2.,np.float32)
        w,stats=temporal_weights(z,self.T,[(z,self.T,self.clear)],self.cam,self.cfg)
        self.assertTrue((w>1).all()); self.assertEqual(stats['conflict'],0)

    def test_occlusion_and_missing_history_are_neutral(self):
        current=np.full((9,9),2.,np.float32)
        old=np.full((9,9),1.,np.float32)
        w,stats=temporal_weights(current,self.T,[(old,self.T,self.clear)],self.cam,self.cfg)
        np.testing.assert_array_equal(w,np.ones_like(w)); self.assertEqual(stats['occluded'],81)
        w,_=temporal_weights(current,self.T,[(old*0,self.T,self.clear)],self.cam,self.cfg)
        np.testing.assert_array_equal(w,np.ones_like(w))

    def test_two_contradictions_reject_phantom_foreground(self):
        current=np.full((9,9),1.,np.float32); old=current*2
        w,stats=temporal_weights(current,self.T,[(old,self.T,self.clear)]*2,self.cam,self.cfg)
        self.assertEqual(stats['rejected'],81); self.assertFalse(w.any())

    def test_previous_depth_edge_does_not_reject(self):
        z=np.ones((9,9),np.float32); old=z*2
        w,_=temporal_weights(z,self.T,[(old,self.T,np.ones_like(self.clear))]*2,self.cam,self.cfg)
        np.testing.assert_array_equal(w,np.ones_like(w))


class VolumeTests(unittest.TestCase):
    def test_surfel_support_counts_views_not_pixels(self):
        from fusion_tsdf import VoxelSurfels
        cam=Camera(9,9,1000,1000,4,4,.1)
        volume=VoxelSurfels([-.1,-.1,1.8],[.1,.1,2.2],cam,.05,device='cpu')
        depth=np.full((9,9),2.,np.float32); weights=np.ones_like(depth)
        volume.integrate(depth,weights,np.zeros_like(depth),np.eye(4))
        self.assertEqual(int(volume.count.max()),1)
        first=volume.count.clone()
        volume.integrate(depth,weights,np.zeros_like(depth),np.eye(4))
        np.testing.assert_array_equal(volume.count.numpy(),(2*first).numpy())
        good=volume.count>0
        np.testing.assert_allclose((volume.xyz[good]/volume.weight[good,None]).numpy()[:,2],2.,atol=1e-6)

    def test_plane_and_weighted_update_match_analytic_value(self):
        from fusion_tsdf import FrustumTSDF
        cam=Camera(9,9,10,10,4,4,.1)
        v=FrustumTSDF([-.1,-.1,1.8],[.1,.1,2.2],cam,.05,.15,device='cpu',chunk=100)
        D=np.full((9,9),2.,np.float32); W=np.ones_like(D); gray=np.zeros_like(D)
        v.integrate(D,W*3,gray,np.eye(4)); v.integrate(D+.05,W,gray,np.eye(4))
        # Interior voxel z=1.975, both observations in truncation band.
        expected=(3*((2-1.975)/.15)+(2.05-1.975)/.15)/4
        self.assertAlmostEqual(float(v.tsdf[1,1,3]),expected,places=5)
        self.assertEqual(int(v.count[1,1,3]),2)
        before=v.tsdf.clone(); count=v.count.clone()
        v.integrate(np.zeros_like(D),W,gray,np.eye(4))
        np.testing.assert_array_equal(before.numpy(),v.tsdf.numpy())
        np.testing.assert_array_equal(count.numpy(),v.count.numpy())


if __name__=='__main__': unittest.main()
