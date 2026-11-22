#!/usr/bin/env python3
"""Make an exported RT-IGEV ONNX buildable with TensorRT 10.3 FP16 (JetPack 6.2, Orin).

    python3 onnx_for_trt.py rtigev_sceneflow_i6_512x640.onnx rtigev_sceneflow_i6_512x640_trt.onnx

1. onnx-simplifier: constant folding, static shapes (3330 -> 931 nodes for 6 GRU iterations).
2. Every Conv whose output feeds more than 8 point-wise ops is also exposed as a graph output. In RT-IGEV this is
   /net/desc/Conv: its 96x128x160 feature map is multiplied 48 times (one per disparity of the correlation volume).
   TensorRT 10.3 tries to fuse the conv with all 48 multiplications into one FP16 kernel and aborts with
   "genericPwFusion ... kMAX_INPUT_COUNT_WITH_JIT"; an output tensor cannot be fused away, which avoids it.
   The extra output is ignored by the depth node.
The disparity output is checked to be identical (onnxruntime, CPU) before writing.

Then on the Jetson:  trtexec --onnx=<out> --fp16 --saveEngine=rtigev_i6_512x640_fp16.engine
Needs: onnx, onnxsim, onnxruntime, numpy (any machine; the result is hardware independent).
"""
import sys

import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
from onnx import helper, shape_inference

POINTWISE = {'Mul', 'Add', 'Sub', 'Div', 'Relu', 'LeakyRelu', 'Sigmoid', 'Tanh', 'Clip'}


def main(src, dst):
    model, ok = onnxsim.simplify(onnx.load(src))
    if not ok:
        raise SystemExit('onnx-simplifier could not validate the simplified model')
    g = model.graph
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    hot = [n.output[0] for n in g.node if n.op_type in ('Conv', 'ConvTranspose')
           and sum(c.op_type in POINTWISE for c in consumers.get(n.output[0], [])) > 8]
    info = {v.name: v for v in shape_inference.infer_shapes(model).graph.value_info}
    for t in hot:
        g.output.append(info[t] if t in info else helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
    onnx.checker.check_model(model)
    print('simplified nodes:', len(g.node), ' extra outputs (fusion barriers):', hot)

    h, w = [d.dim_value for d in g.input[0].type.tensor_type.shape.dim][2:]
    rng = np.random.default_rng(0)
    left = (rng.random((1, 1, h, w)) * 255).astype(np.float32)
    right = np.roll(left, -12, axis=3)
    a = ort.InferenceSession(src, providers=['CPUExecutionProvider']).run(['disparity'], {'left': left, 'right': right})[0]
    b = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider']).run(
        ['disparity'], {'left': left, 'right': right})[0]
    diff = float(np.abs(a - b).max())
    print(f'disparity max |difference| vs the input model: {diff:.2e}')
    if diff > 1e-3:
        raise SystemExit('outputs differ, not writing')
    onnx.save(model, dst)
    print('wrote', dst)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
