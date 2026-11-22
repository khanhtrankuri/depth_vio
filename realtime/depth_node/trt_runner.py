"""Stereo network runners without PyTorch.

TrtStereo  : TensorRT engine (JetPack's `tensorrt` Python package). CUDA memory, copies and the optional CUDA-graph
             replay go through libcudart via ctypes, so neither pycuda nor cuda-python is needed.
OnnxStereo : onnxruntime fallback (CPU or CUDA EP) for testing the node on a machine without TensorRT.

Both take rectified float32 HxW images in [0,255] and return the HxW float32 left disparity [px].
"""
import ctypes
import ctypes.util
import glob

import numpy as np

H2D, D2H = 1, 2
CAPTURE_MODE_THREAD_LOCAL = 1


def _load_cudart():
    names = [ctypes.util.find_library('cudart')] + sorted(glob.glob('/usr/local/cuda*/lib64/libcudart.so*'))
    for n in names:
        if n:
            try:
                return ctypes.CDLL(n)
            except OSError:
                pass
    raise OSError('libcudart not found (is CUDA installed? try /usr/local/cuda/lib64)')


class _Cuda:
    def __init__(self):
        self.lib = _load_cudart()

    def check(self, err, what):
        if err != 0:
            self.lib.cudaGetErrorString.restype = ctypes.c_char_p
            raise RuntimeError(f'{what} failed: {self.lib.cudaGetErrorString(err).decode()} ({err})')

    def malloc(self, nbytes):
        p = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(nbytes)), 'cudaMalloc')
        return p

    def stream(self):
        s = ctypes.c_void_p()
        self.check(self.lib.cudaStreamCreate(ctypes.byref(s)), 'cudaStreamCreate')
        return s

    def copy_async(self, dst, src, nbytes, kind, stream):
        self.check(self.lib.cudaMemcpyAsync(dst, src, ctypes.c_size_t(nbytes), ctypes.c_int(kind), stream),
                   'cudaMemcpyAsync')

    def sync(self, stream):
        self.check(self.lib.cudaStreamSynchronize(stream), 'cudaStreamSynchronize')


class TrtStereo:
    def __init__(self, engine_path, cuda_graph=True, logger=None):
        import tensorrt as trt
        self.cuda = _Cuda()
        log = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(log) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f'could not deserialize {engine_path} (built with another TensorRT version/GPU?)')
        self.ctx = self.engine.create_execution_context()
        self.stream = self.cuda.stream()
        self.host, self.dev, self.inputs, self.outputs = {}, {}, [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(s < 0 for s in shape):
                raise RuntimeError(f'{name} has a dynamic shape {shape}; build the engine with fixed shapes')
            arr = np.zeros(shape, trt.nptype(self.engine.get_tensor_dtype(name)))
            self.host[name] = arr
            self.dev[name] = self.cuda.malloc(arr.nbytes)
            self.ctx.set_tensor_address(name, self.dev[name].value)
            (self.inputs if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else self.outputs).append(name)
        if len(self.inputs) != 2 or 'disparity' not in self.outputs:
            raise RuntimeError(f'expected inputs (left, right) and output disparity, got {self.inputs} -> {self.outputs}')
        self.left, self.right = ('left', 'right') if 'left' in self.inputs else self.inputs
        self.size = self.host[self.left].shape[-2:]
        self.graph_exec = None
        self._enqueue()  # first run initialises TensorRT's lazily created resources
        self.cuda.sync(self.stream)
        if cuda_graph:
            try:
                self._capture()
            except Exception as e:  # noqa: BLE001  (graph replay is only an optimisation)
                self.graph_exec = None
                if logger:
                    logger(f'CUDA graph capture failed, using plain enqueue: {e}')

    def _enqueue(self):
        if not self.ctx.execute_async_v3(self.stream.value):
            raise RuntimeError('TensorRT execute_async_v3 failed')

    def _capture(self):
        lib = self.cuda.lib
        graph, exe = ctypes.c_void_p(), ctypes.c_void_p()
        self.cuda.check(lib.cudaStreamBeginCapture(self.stream, ctypes.c_int(CAPTURE_MODE_THREAD_LOCAL)),
                        'cudaStreamBeginCapture')
        self._enqueue()
        self.cuda.check(lib.cudaStreamEndCapture(self.stream, ctypes.byref(graph)), 'cudaStreamEndCapture')
        self.cuda.check(lib.cudaGraphInstantiate(ctypes.byref(exe), graph, ctypes.c_ulonglong(0)),
                        'cudaGraphInstantiate')
        self.graph_exec = exe

    def __call__(self, left, right):
        h, w = self.size
        if left.shape != (h, w):
            raise ValueError(f'engine expects {h}x{w} images, got {left.shape[0]}x{left.shape[1]}')
        for name, img in ((self.left, left), (self.right, right)):
            buf = self.host[name]
            buf[0, 0] = img  # casts to the engine input dtype
            self.cuda.copy_async(self.dev[name], buf.ctypes.data_as(ctypes.c_void_p), buf.nbytes, H2D, self.stream)
        if self.graph_exec is not None:
            self.cuda.check(self.cuda.lib.cudaGraphLaunch(self.graph_exec, self.stream), 'cudaGraphLaunch')
        else:
            self._enqueue()
        out = self.host['disparity']
        self.cuda.copy_async(out.ctypes.data_as(ctypes.c_void_p), self.dev['disparity'], out.nbytes, D2H, self.stream)
        self.cuda.sync(self.stream)
        return out[0, 0].astype(np.float32, copy=False)


class OnnxStereo:
    def __init__(self, onnx_path, providers=None):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(onnx_path, providers=providers or ort.get_available_providers())
        shp = self.sess.get_inputs()[0].shape
        self.size = (int(shp[2]), int(shp[3]))

    def __call__(self, left, right):
        out = self.sess.run(['disparity'], {'left': left[None, None].astype(np.float32),
                                            'right': right[None, None].astype(np.float32)})[0]
        return out[0, 0]
