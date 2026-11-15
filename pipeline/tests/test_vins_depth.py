"""Physical geometry and evaluation controls for the depth-in-VINS extension."""
import sys
from pathlib import Path
import unittest
import tempfile
import struct
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from export_igev_vins_depth import raw_inverse_map
from evaluate_vins_depth import scores
from prepare_depth_sweep import select_exposures,scale_binary


class DepthSweepControls(unittest.TestCase):
    def test_noise_weight_changes_only_sigma(self):
        with tempfile.TemporaryDirectory() as directory:
            source,target=Path(directory)/'source.bin',Path(directory)/'scaled.bin'
            values=np.array([[.5,.04],[0,0],[1,.1],[.25,.03]],dtype='<f4')
            source.write_bytes(struct.pack('<II',2,2)+values.tobytes())
            scale_binary(source,target,2.)
            scaled=np.frombuffer(target.read_bytes()[8:],dtype='<f4').reshape(-1,2)
            np.testing.assert_array_equal(scaled[:,0],values[:,0])
            np.testing.assert_allclose(scaled[:,1],values[:,1]*2)
            np.testing.assert_array_equal(np.frombuffer(source.read_bytes()[8:],dtype='<f4').reshape(-1,2),values)

    def test_frequency_selection_preserves_backend_exposure_phase(self):
        # Camera at 20Hz; depth cache every 5 images; VINS consumes odd image indices.
        times=list(range(0,2000000000,50000000))
        index=[(t,str(t)+'.bin') for t in times[::5]]
        full=select_exposures(index,times,2.)
        self.assertEqual([t for t,_ in full],[250000000,750000000,1250000000,1750000000])
        self.assertEqual(select_exposures(index,times,1.),full[::2])
        self.assertEqual(select_exposures(index,times,.5),full[::4])


class DepthGeometryTests(unittest.TestCase):
    def test_rectified_z_is_converted_to_raw_optical_z(self):
        K=np.array([[100.,0,10],[0,100.,10],[0,0,1.]])
        R=Rotation.from_euler('y',3,degrees=True).as_matrix()
        rect=dict(R0=R,fx=100.,fy=100.,cx=10.,cy=10.,baseline=.1)
        d=np.full((21,21),5.,np.float32)
        result=raw_inverse_map(d,np.ones_like(d,bool),rect,K,np.zeros(4))
        self.assertAlmostEqual(float(result[10,10,0]),np.cos(np.deg2rad(3))*.5,places=6)
        self.assertNotAlmostEqual(float(result[10,10,0]),.5,places=5)
        self.assertGreater(float(result[10,10,1]),0)

    def test_invalid_depth_and_out_of_range_are_not_constraints(self):
        K=np.array([[100.,0,2],[0,100.,2],[0,0,1.]])
        rect=dict(R0=np.eye(3),fx=100.,fy=100.,cx=2.,cy=2.,baseline=.1)
        for d,mask in [(np.full((5,5),np.nan),np.ones((5,5),bool)),
                       (np.full((5,5),.1),np.ones((5,5),bool)),
                       (np.full((5,5),5.),np.zeros((5,5),bool))]:
            np.testing.assert_array_equal(raw_inverse_map(d,mask,rect,K,np.zeros(4)),0)


class MetricControls(unittest.TestCase):
    def test_global_rigid_transform_does_not_create_ate_or_rpe(self):
        t=np.arange(101,dtype=np.int64)*100000000
        s=np.arange(101)*.1
        P=np.column_stack([np.sin(s),np.cos(s),s*.1])
        Q=Rotation.from_euler('z',(s*.1)[:,None]).as_matrix()
        rotation=Rotation.from_euler('xyz',[.2,-.1,.5]).as_matrix()
        G=P@rotation.T+[3,2,-1];GQ=np.einsum('ij,njk->nik',rotation,Q)
        result,_=scores(t,P,Q,G,GQ)
        self.assertLess(result['ATE_RMSE_m'],1e-12)
        self.assertLess(result['RPE_1s']['translation_RMSE_m'],1e-12)
        self.assertLess(result['RPE_1s']['rotation_RMSE_deg'],1e-10)
        # SE3 fitting must not absorb a metric scale error.
        result,_=scores(t,P*1.2,Q,G,GQ)
        self.assertGreater(result['ATE_RMSE_m'],.05)


if __name__=='__main__':unittest.main()
