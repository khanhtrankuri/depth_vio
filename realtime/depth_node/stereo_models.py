"""Learned stereo models for the live depth node (official IGEV++ / RT-IGEV code, pretrained weights).

Every model maps rectified grayscale float tensors (1,1,H,W) in [0,255] to a left disparity tensor (H,W).
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
IGEV_DIR = Path(os.environ.get('IGEV_DIR', ROOT / 'IGEV-plusplus'))
WEIGHTS = IGEV_DIR / 'pretrained_models'


def _import_igev():
    if str(IGEV_DIR) not in sys.path:
        sys.path.insert(0, str(IGEV_DIR))
    import timm
    if not getattr(timm, '_depth_vio_patched', False):
        create = timm.create_model
        # the backbone weights are overwritten by the stereo checkpoint: never download ImageNet weights
        timm.create_model = lambda *a, **k: create(*a, **{**k, 'pretrained': False})
        timm._depth_vio_patched = True


def _load(net, ckpt):
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    state = {k.replace('module.', '', 1): v for k, v in state.items()}
    net.load_state_dict(state, strict=True)
    return net


class StereoModel:
    """Wraps a stereo network; optionally replays the whole forward pass as one CUDA graph.

    The graph removes per-kernel launch overhead (>1000 small kernels per frame), which dominates
    latency under WSL2 and on embedded GPUs. Input shape must stay fixed once captured.
    """

    def __init__(self, net, iters, divis_by=32, half=True, cuda_graph=True):
        self.net = net.cuda().eval()
        self.iters = iters
        self.divis_by = divis_by
        self.half = half
        self.cuda_graph = cuda_graph
        self._graph = None

    def _forward(self, left, right):
        h, w = left.shape[-2:]
        ph = (-h) % self.divis_by
        pw = (-w) % self.divis_by
        pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
        a = F.pad(left, pad, mode='replicate').expand(-1, 3, -1, -1)
        b = F.pad(right, pad, mode='replicate').expand(-1, 3, -1, -1)
        d = self.net(a, b, iters=self.iters, test_mode=True)
        d = d[..., pad[2]:pad[2] + h, pad[0]:pad[0] + w]
        return d[0, 0].float()

    @torch.inference_mode()
    def __call__(self, left, right):
        if not self.cuda_graph:
            return self._forward(left, right)
        if self._graph is None or self._in[0].shape != left.shape:
            self._in = (left.clone(), right.clone())
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._forward(*self._in)
            torch.cuda.current_stream().wait_stream(s)
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                self._out = self._forward(*self._in)
        self._in[0].copy_(left)
        self._in[1].copy_(right)
        self._graph.replay()
        return self._out.clone()


def _patch_rt_geometry():
    """Same math as core_rt.geometry.Geo_Encoding_Volume.__call__, but builds the offset tensor on the
    GPU (the original creates it on the CPU and copies it, which is illegal inside CUDA-graph capture)."""
    import core_rt.geometry as g
    if getattr(g.Geo_Encoding_Volume, '_depth_vio_patched', False):
        return

    def call(self, disp):
        r = self.radius
        b, _, h, w = disp.shape
        dx = torch.linspace(-r, r, 2 * r + 1, device=disp.device, dtype=disp.dtype).view(1, 1, 2 * r + 1, 1)
        out = []
        for i in range(self.num_levels):
            x0 = dx + disp.reshape(b * h * w, 1, 1, 1) / 2 ** i
            vol_i = self.geo_volume_pyramid[i]  # (bhw, c, 1, d): 1-D sampling along disparity, y == 0
            xg = 2 * x0 / (vol_i.shape[-1] - 1) - 1
            vol = F.grid_sample(vol_i, torch.cat([xg, torch.zeros_like(xg)], dim=-1), align_corners=True)
            out.append(vol.view(b, h, w, -1))
        return torch.cat(out, dim=-1).permute(0, 3, 1, 2).contiguous().float()
    g.Geo_Encoding_Volume.__call__ = call
    g.Geo_Encoding_Volume._depth_vio_patched = True


def rt_igev(weights='sceneflow', iters=6, half=True):
    _import_igev()
    _patch_rt_geometry()
    from core_rt.rt_igev_stereo import IGEVStereo
    a = argparse.Namespace(mixed_precision=half, precision_dtype='float16', hidden_dim=96, corr_levels=2,
                           corr_radius=4, n_downsample=2, n_gru_layers=3, max_disp=192)
    return StereoModel(_load(IGEVStereo(a), WEIGHTS / 'igev_rt' / f'{weights}.pth'), iters, half=half)


def igev_pp(weights='eth3d', iters=8, half=True):
    _import_igev()
    from core.igev_stereo import IGEVStereo
    a = argparse.Namespace(mixed_precision=half, precision_dtype='float16', hidden_dims=[128] * 3, corr_levels=2,
                           corr_radius=4, n_downsample=2, n_gru_layers=3, max_disp=768, s_disp_range=48,
                           m_disp_range=96, l_disp_range=192, s_disp_interval=1, m_disp_interval=2,
                           l_disp_interval=4)
    # IGEV++ creates CPU tensors inside forward(): no CUDA-graph capture (too slow for live use anyway).
    return StereoModel(_load(IGEVStereo(a), WEIGHTS / 'igev_plusplus' / f'{weights}.pth'), iters, half=half,
                       cuda_graph=False)


def build(spec):
    """spec examples: 'rt:sceneflow:6', 'pp:eth3d:8'."""
    kind, weights, iters = spec.split(':')
    return {'rt': rt_igev, 'pp': igev_pp}[kind](weights, int(iters))
