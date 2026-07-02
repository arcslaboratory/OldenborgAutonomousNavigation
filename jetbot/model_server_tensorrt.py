import io
import json
import pickle
import zipfile
from argparse import ArgumentParser

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from rpc import RPCServer


# ---------------------------------------------------------------------------
# Camera calibration constants
# ---------------------------------------------------------------------------
MTX = np.array([[103.80088948, 0.0, 112.99926554],
                [0.0, 133.49980795, 99.02992825],
                [0.0, 0.0, 1.0]])
DIST = np.array([[-0.2964748, 0.07032658, 0.00985859, -0.00041996, -0.00725899]])


def apply_corrections(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("Received None image")

    # 1. Undistort
    h, w = img.shape[:2]
    new_mtx, _ = cv2.getOptimalNewCameraMatrix(MTX, DIST, (w, h), 0, (w, h))
    undistorted = cv2.undistort(img, MTX, DIST, None, new_mtx)

    # 2. Crop top half, then resize back to original dimensions
    top_half = undistorted[0 : h // 2, 0:w]
    undistorted = cv2.resize(top_half, (w, h), interpolation=cv2.INTER_LINEAR)

    # 3. Gray-world correction
    result = undistorted.astype(np.float32)
    result[:, :, 0] *= 0.88
    result[:, :, 1] *= 1.08
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = ArgumentParser(description="TensorRT model server")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint (.pt/.pth) or label map (.json). "
                             "Used only to recover the idx_to_class mapping.")
    parser.add_argument("--engine", type=str, required=True,
                        help="Path to the TensorRT .engine file")
    parser.add_argument("--rpc_host", type=str, default="127.0.0.1")
    parser.add_argument("--rpc_port", type=int, default=8033)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Global state (set once in main, used in model_run)
# ---------------------------------------------------------------------------
_context = None
_d_input = None
_d_output = None
_stream = None
_h_output = None
_idx_to_class = None
_cuda_context = None  # explicit CUDA context shared by pycuda and TensorRT


def _dummy(*args, **kwargs):
    """Placeholder for any torch tensor reconstruction we don't need."""
    return None


class TorchUnpickler(pickle.Unpickler):
    """
    Custom unpickler that handles PyTorch's zip-format checkpoints without
    needing to reconstruct tensors. We only need plain Python objects
    (dicts, strings, ints) so all tensor/storage references return None.
    """
    _DUMMY_MODULES = {"torch", "torch._utils", "torch.storage"}

    def persistent_load(self, pid):
        return None

    def find_class(self, module, name):
        if module in self._DUMMY_MODULES or name.startswith("_rebuild_tensor"):
            return _dummy
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return _dummy


def load_idx_to_class(checkpoint_path):
    """
    Load the idx_to_class mapping either from a JSON file (preferred) or
    from a PyTorch checkpoint (.pt/.pth).
    """
    if checkpoint_path.endswith(".json"):
        with open(checkpoint_path, "r") as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

    if zipfile.is_zipfile(checkpoint_path):
        with zipfile.ZipFile(checkpoint_path, "r") as zf:
            with zf.open("archive/data.pkl") as pkl_file:
                data = pkl_file.read()
        ckpt = TorchUnpickler(io.BytesIO(data)).load()
    else:
        import torch
        ckpt = torch.load(checkpoint_path, map_location="cpu")

    return {int(k): v for k, v in ckpt["idx_to_class"].items()}


def model_run(image_filename):
    global _context, _d_input, _d_output, _stream, _h_output, _idx_to_class
    global _cuda_context

    # Push the shared CUDA context before any CUDA work
    _cuda_context.push()

    try:
        # 1. Load and preprocess
        img_bgr = cv2.imread(str(image_filename))
        img_bgr = apply_corrections(img_bgr)

        img = cv2.resize(img_bgr, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        h_input = np.transpose(img, (2, 0, 1)).reshape(1, 3, 224, 224).astype(np.float32)
        h_input = np.ascontiguousarray(h_input)

        # 2. Inference
        cuda.memcpy_htod_async(_d_input, h_input, _stream)
        _context.execute_async(batch_size=1, bindings=[int(_d_input), int(_d_output)], stream_handle=_stream.handle)
        cuda.memcpy_dtoh_async(_h_output, _d_output, _stream)
        _stream.synchronize()

        # 3. Softmax
        logits = _h_output
        probs = np.exp(logits - np.max(logits)) / np.sum(np.exp(logits - np.max(logits)))

        pred_class = _idx_to_class[int(np.argmax(probs))]
        print("Command: {} | forward={:.2f} left={:.2f} right={:.2f}".format(
            pred_class, probs[0], probs[1], probs[2]))
        return pred_class

    finally:
        # Always pop the context, even if inference throws
        _cuda_context.pop()


def main():
    global _context, _d_input, _d_output, _stream, _h_output, _idx_to_class
    global _cuda_context

    args = parse_args()

    _idx_to_class = load_idx_to_class(args.checkpoint)
    print("Loaded idx_to_class:", _idx_to_class)

    # Initialize CUDA explicitly — do NOT use pycuda.autoinit.
    # autoinit creates its own context which conflicts with TensorRT's context.
    # Instead we create one context and share it between pycuda and TensorRT.
    cuda.init()
    device = cuda.Device(0)
    _cuda_context = device.make_context()

    try:
        # Initialize TensorRT inside the shared CUDA context
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(args.engine, "rb") as f, trt.Runtime(trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        _context = engine.create_execution_context()

        # Allocate GPU memory
        _d_input = cuda.mem_alloc(1 * 3 * 224 * 224 * 4)
        _d_output = cuda.mem_alloc(3 * 4)  # 3 output classes
        _h_output = cuda.pagelocked_empty(3, dtype=np.float32)
        _stream = cuda.Stream()

        # Pop context here — model_run will push/pop it per call
        _cuda_context.pop()

        server = RPCServer(args.rpc_host, args.rpc_port)
        server.registerMethod(model_run)
        print("Server running on {}:{}".format(args.rpc_host, args.rpc_port))
        server.run()

    except Exception as e:
        _cuda_context.pop()
        raise e


if __name__ == "__main__":
    main()