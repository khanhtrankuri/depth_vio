"""Thin wrapper around the official IGEV++ code (gangweiX/IGEV-plusplus) for inference."""
import os, sys, argparse
import numpy as np
import torch
from common import ROOT

IGEV_DIR = os.path.join(ROOT, "IGEV-plusplus")
sys.path.insert(0, IGEV_DIR); sys.path.insert(0, os.path.join(IGEV_DIR, "core"))
from core.igev_stereo import IGEVStereo  # noqa: E402
from core.utils.utils import InputPadder  # noqa: E402

CKPT = os.environ.get("IGEV_CKPT", os.path.join(IGEV_DIR, "pretrained_models", "igev_plusplus", "sceneflow.pth"))


def make_args():
    return argparse.Namespace(
        restore_ckpt=CKPT, mixed_precision=True, precision_dtype="float16", valid_iters=16,
        hidden_dims=[128] * 3, corr_levels=2, corr_radius=4, n_downsample=2, n_gru_layers=3,
        max_disp=768, s_disp_range=48, m_disp_range=96, l_disp_range=192,
        s_disp_interval=1, m_disp_interval=2, l_disp_interval=4)


class IGEV:
    def __init__(self, ckpt=CKPT, iters=16):
        a = make_args(); a.restore_ckpt = ckpt
        net = torch.nn.DataParallel(IGEVStereo(a), device_ids=[0])
        net.load_state_dict(torch.load(ckpt, map_location="cuda", weights_only=False))
        self.net = net.module.cuda().eval()
        self.iters = iters
        self.ckpt = ckpt

    @torch.no_grad()
    def __call__(self, left_u8, right_u8):
        """Gray uint8 HxW rectified images → disparity (float32 HxW, px, left view)."""
        def t(img):
            x = torch.from_numpy(np.repeat(img[..., None], 3, -1)).permute(2, 0, 1).float()
            return x[None].cuda()
        a, b = t(left_u8), t(right_u8)
        pad = InputPadder(a.shape, divis_by=32)
        a, b = pad.pad(a, b)
        d = self.net(a, b, iters=self.iters, test_mode=True)
        return pad.unpad(d).float().cpu().numpy().squeeze()

    def right_disp(self, left_u8, right_u8):
        """Disparity in the right view via horizontal flip trick (for L-R consistency)."""
        return self(np.ascontiguousarray(right_u8[:, ::-1]), np.ascontiguousarray(left_u8[:, ::-1]))[:, ::-1]


def lr_check(dL, dR, thresh=1.0):
    """Left-right consistency mask: |dL(u) − dR(u − dL(u))| < thresh."""
    h, w = dL.shape
    u = np.arange(w)[None, :].repeat(h, 0)
    ur = np.round(u - dL).astype(int)
    valid = (ur >= 0) & (ur < w) & (dL > 0)
    ur = np.clip(ur, 0, w - 1)
    dr = dR[np.arange(h)[:, None], ur]
    return valid & (np.abs(dL - dr) < thresh)
