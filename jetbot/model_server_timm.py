from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm
import tyro
from rpc import RPCServer
import cv2.xphoto

# ---------------------------------------------------------------------------
# Camera calibration constants (224x224 recalibrated)
# ---------------------------------------------------------------------------

MTX = np.array([[103.80088948,   0.0,         112.99926554],
                [  0.0,         133.49980795,  99.02992825],
                [  0.0,           0.0,           1.0        ]])

DIST = np.array([[-0.2964748, 0.07032658, 0.00985859, -0.00041996, -0.00725899]])


def apply_corrections(img: np.ndarray) -> np.ndarray:
    """Undistort + gray-world color correction (BGR uint8 → BGR uint8).

    Matches the dataset preprocessing script exactly:
      1. Geometric undistortion with alpha=0 (zoom in, no black edges)
      2. Gray-world color correction per channel
    The robot client saves raw uncorrected JPEGs so this function sees the
    same input distribution as the dataset script did.
    """
    if img is None:
        raise ValueError("apply_corrections received None image")

    # 1. Geometric undistortion
    h, w = img.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(MTX, DIST, (w, h), 0, (w, h))
    undistorted = cv2.undistort(img, MTX, DIST, None, new_camera_mtx)

    # 2. Gray-world color correction
    result = undistorted.astype(np.float32)
    avg_b, avg_g, avg_r = np.mean(result, axis=(0, 1))
    avg_gray = (avg_b + avg_g + avg_r) / 3
    
#     result[:, :, 0] *= avg_gray / avg_b if avg_b > 0 else 1.0
#      result[:, :, 0] *= avg_gray / avg_b if avg_b > 0 else 1.0
#          result[:, :, 0] *= avg_gray / avg_b if avg_b > 0 else 1.0
    result[:, :, 0] *= 0.88   # Blue suppress
    result[:, :, 1] *= 1.08   # Green boost
    result[:, :, 2] *= 1.00   # Red unchanged

    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Validation transform (must match train_line_model.py exactly)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_val_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

@dataclass
class Args:
    checkpoint: str                        # path to best_model.pth
    device: str = ""                       # "cuda" / "cpu" (auto if empty)
    rpc_host: str = "127.0.0.1"
    rpc_port: int = 8033


# ---------------------------------------------------------------------------
# Global model state (set in main, used by model_run)
# ---------------------------------------------------------------------------

_model = None
_transform = None
_idx_to_class: dict = {}
_device = None


#DEBUG FUNCTION FOR COLOR CALIBRATION

# def diagnose_color(img_bgr: np.ndarray):
#     """Prints exact channel averages to help tune color correction."""
#     img = img_bgr.astype(np.float32)
    
#     avg_b = np.mean(img[:, :, 0])
#     avg_g = np.mean(img[:, :, 1])
#     avg_r = np.mean(img[:, :, 2])
    
#     print(f"Raw channel averages:")
#     print(f"  B: {avg_b:.2f}")
#     print(f"  G: {avg_g:.2f}")
#     print(f"  R: {avg_r:.2f}")
#     print(f"  B/G ratio: {avg_b/avg_g:.3f}  (>1.0 = too blue)")
#     print(f"  R/G ratio: {avg_r/avg_g:.3f}  (<1.0 = not enough red)")
    
#     avg_gray = (avg_b + avg_g + avg_r) / 3.0
#     print(f"Needed gains:")
#     print(f"  gain_b: {avg_gray/avg_b:.3f}")
#     print(f"  gain_g: {avg_gray/avg_g:.3f}")
#     print(f"  gain_r: {avg_gray/avg_r:.3f}")


def model_run(image_filename: str) -> str:
    """
    Called by the RPC client with a path to a saved PNG/JPG.

    Pipeline:
      1. Load image with OpenCV (BGR)
      2. Apply undistort + gray-world correction
      3. Convert BGR → RGB PIL Image
      4. Apply validation transform (resize / crop / normalize)
      5. Run model forward pass
      6. Return predicted class string ("left" / "forward" / "right")
    """
    # 1. Load
    img_bgr = cv2.imread(str(image_filename))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_filename}")
        

#     print("BEFORE correction:")
#     diagnose_color(img_bgr)

    # 2. Correct
    img_bgr = apply_corrections(img_bgr)

#     print("AFTER correction:")
#     diagnose_color(img_bgr)


    # 3. BGR → RGB PIL
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # 4. Transform
    tensor = _transform(pil_img).unsqueeze(0).to(_device)  # [1, C, H, W]

    # 5. Inference
    with torch.no_grad():
        logits = _model(tensor)
        probs  = F.softmax(logits, dim=1)[0]

    pred_idx   = probs.argmax().item()
    pred_class = _idx_to_class[pred_idx]
    pred_prob  = probs[pred_idx].item()

    print(f"  → {pred_class}  ({pred_prob:.3f})")
    return pred_class


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _model, _transform, _idx_to_class, _device

    args = tyro.cli(Args)

#     Device
    if args.device:
        _device = torch.device(args.device)
    else:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {_device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=_device)
    model_name   = ckpt["model_name"]
    img_size     = ckpt["img_size"]
    _idx_to_class = {int(k): v for k, v in ckpt["idx_to_class"].items()}
    n_classes    = len(_idx_to_class)

    print(f"Model      : {model_name}")
    print(f"Image size : {img_size}")
    print(f"Classes    : {_idx_to_class}")

    # Build model
    _model = timm.create_model(model_name, pretrained=False, num_classes=n_classes)
    _model.load_state_dict(ckpt["model_state_dict"])
    _model = _model.to(_device)
    _model.eval()

    # Build transform
    _transform = build_val_transform(img_size)

    # Start RPC server
    server = RPCServer(args.rpc_host, args.rpc_port)
    server.registerMethod(model_run)
    print(f"RPC server listening on {args.rpc_host}:{args.rpc_port}")
    server.run()


if __name__ == "__main__":
    main()