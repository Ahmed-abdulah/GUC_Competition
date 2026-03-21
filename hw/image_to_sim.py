# =============================================================
#  image_to_sim.py
#  Converts a real handwritten digit image into:
#    1. A hex file for Vivado simulation testbench
#    2. SW model prediction (ground truth to compare against)
#
#  Usage:
#    python image_to_sim.py --image my_digit.png --label 6
#    python image_to_sim.py --image my_digit.png  (label optional)
#
#  Output files:
#    sim_image.hex     → loaded by testbench via $readmemh
#    sim_image.png     → preview of preprocessed image
# =============================================================

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# ── LeNet-5 model (must match training script) ─────────────
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1  = nn.Conv2d(1, 6, 5)
        self.pool1  = nn.MaxPool2d(2, 2)
        self.conv2  = nn.Conv2d(6, 16, 5)
        self.pool2  = nn.MaxPool2d(2, 2)
        self.fc1    = nn.Linear(256, 120)
        self.fc2    = nn.Linear(120, 84)
        self.fc3    = nn.Linear(84, 10)
        self.relu   = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# ── Fixed-point parameters (must match FPGA) ───────────────
FRAC_BITS = 6
SCALE     = 2 ** FRAC_BITS   # 64
MAX_VAL   = 127
MIN_VAL   = -128

def preprocess_image(img_path, preview=True):
    """
    Load and preprocess a handwritten digit image for FPGA.
    Steps:
      1. Load image, convert to grayscale
      2. Invert if background is white (MNIST = white digit, black bg)
      3. Resize to 28x28
      4. Normalize using MNIST statistics
      5. Convert to 8-bit unsigned (0-255) for UART transmission
    """
    img = Image.open(img_path).convert('L')  # grayscale

    # Auto-invert: MNIST has white digit on black background.
    # If mean pixel > 128, the image is light-on-dark → invert.
    img_arr = np.array(img, dtype=np.float32)
    if img_arr.mean() > 128:
        img = ImageOps.invert(img)
        print("  Auto-inverted image (was light background)")

    # Resize to 28×28
    img = img.resize((28, 28), Image.LANCZOS)
    img_arr = np.array(img, dtype=np.float32)

    # Save preview
    if preview:
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(np.array(Image.open(img_path).convert('L')),
                       cmap='gray', vmin=0, vmax=255)
        axes[0].set_title("Original")
        axes[0].axis('off')
        axes[1].imshow(img_arr, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title("Preprocessed (28×28)")
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig("sim_image_preview.png", dpi=100)
        plt.close()
        print("  Preview saved: sim_image_preview.png")

    return img_arr  # shape (28,28), values 0..255

def image_to_hex(img_arr, out_path="sim_image.hex"):
    """
    Convert preprocessed image to hex file for $readmemh.
    Applies MNIST normalization + fixed-point quantization (FRAC_BITS=6)
    so FPGA input matches the trained model's expected domain.
    Each line = one byte (signed int8 as two's complement, 0x00-0xFF).
    """
    flat = img_arr.flatten().astype(np.float64)
    assert len(flat) == 784, f"Expected 784 pixels, got {len(flat)}"

    # MNIST normalization (same as PyTorch ToTensor + Normalize)
    x_norm = (flat / 255.0 - 0.1307) / 0.3081
    # Quantize to Q6 format (matches FPGA FRAC_BITS)
    x_int = np.round(x_norm * SCALE).astype(np.int32)
    x_int = np.clip(x_int, MIN_VAL, MAX_VAL)
    pixels = (x_int.astype(np.int32) & 0xFF).tolist()

    with open(out_path, 'w') as f:
        for p in pixels:
            f.write(f"{p:02X}\n")

    print(f"  Hex file saved: {out_path}  ({len(pixels)} bytes, MNIST-norm + Q{FRAC_BITS})")
    return pixels

def sw_predict(img_arr, model_path="lenet5_trained.pth"):
    """
    Run SW model prediction on the image.
    This is our ground truth to compare against FPGA output.
    """
    if not os.path.exists(model_path):
        print(f"  WARNING: {model_path} not found — skipping SW prediction")
        return None

    # Normalize using MNIST statistics
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,))
    ])

    img_pil   = Image.fromarray(img_arr.astype(np.uint8))
    tensor    = transform(img_pil).unsqueeze(0)  # [1,1,28,28]

    model = LeNet5()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze()
        pred   = logits.argmax(dim=1).item()

    print(f"\n  SW Model Prediction:")
    print(f"  +-----------------------------+")
    for i in range(10):
        bar = '#' * int(probs[i].item() * 20)
        marker = ' <-- PREDICTED' if i == pred else ''
        print(f"  | Digit {i}: {probs[i]:.4f} {bar}{marker}")
    print(f"  +-----------------------------+")
    print(f"  Predicted digit: {pred}  (confidence: {probs[pred]:.1%})")

    return pred

def generate_testbench_params(img_arr, true_label=None):
    """Print parameters needed in the testbench."""
    flat = img_arr.flatten().astype(np.float64)
    x_norm = (flat / 255.0 - 0.1307) / 0.3081
    x_int = np.clip(np.round(x_norm * SCALE).astype(np.int32), MIN_VAL, MAX_VAL)
    pixels = (x_int.astype(np.int32) & 0xFF).tolist()
    print(f"\n  Testbench parameters:")
    print(f"  FILE: sim_image.hex")
    print(f"  PIXELS: 784")
    if true_label is not None:
        print(f"  EXPECTED CLASS: {true_label}")
    print(f"\n  First 8 pixels (hex): "
          + " ".join(f"{p:02X}" for p in pixels[:8]))

# ── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert handwritten digit image for FPGA simulation"
    )
    parser.add_argument('--image', required=True,
                        help='Path to handwritten digit image (PNG/JPG)')
    parser.add_argument('--label', type=int, default=None,
                        help='True digit label (optional, for verification)')
    parser.add_argument('--output', default='sim_image.hex',
                        help='Output hex file name (default: sim_image.hex)')
    parser.add_argument('--model', default='lenet5_trained.pth',
                        help='Path to trained model')
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  Image -> Simulation Converter")
    print(f"  Input:  {args.image}")
    if args.label is not None:
        print(f"  Label:  {args.label}")
    print(f"{'='*50}\n")

    # Step 1: preprocess image
    print("Step 1: Preprocessing image...")
    img_arr = preprocess_image(args.image)

    # Step 2: save hex file
    print("\nStep 2: Generating hex file for testbench...")
    pixels = image_to_hex(img_arr, args.output)

    # Step 3: SW model prediction
    print("\nStep 3: Running SW model prediction...")
    sw_pred = sw_predict(img_arr, args.model)

    # Step 4: summary
    generate_testbench_params(img_arr, args.label)

    print(f"\n{'='*50}")
    print(f"  READY FOR SIMULATION")
    print(f"  Copy sim_image.hex to your Vivado project folder")
    print(f"  SW model predicts: {sw_pred}")
    if args.label is not None:
        print(f"  True label:        {args.label}")
        if sw_pred is not None:
            status = "CORRECT" if sw_pred == args.label else "WRONG"
            print(f"  SW accuracy:       {status}")
    print(f"  Now run Vivado simulation with tb_lenet5_top.v")
    print(f"  FPGA should output the same digit as SW: {sw_pred}")
    print(f"{'='*50}\n")
