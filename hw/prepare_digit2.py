"""
prepare_digit2.py
─────────────────
ONE script that does everything:
  1. Converts 2.png  →  hw/sim_image.hex
  2. Verifies the hex is not blank
  3. Runs integer reference inference on it
  4. Prints FC3 logits + predicted class

Run from your GUC/ folder:
    python prepare_digit2.py

Requirements: pip install pillow numpy
"""

import os, sys, struct
import numpy as np
from PIL import Image

# ── CONFIG ────────────────────────────────────────────────────────────────────
IMG_PATH    = sys.argv[1] if len(sys.argv) > 1 else "2.png"
HEX_OUT     = "sim_image.hex"
WEIGHTS_DIR = "hex_weights"

FRAC_BITS   = 6                         # Q6 fixed point
SCALE       = 1 << FRAC_BITS           # 64
# MNIST normalization constants
MNIST_MEAN  = 0.1307
MNIST_STD   = 0.3081
# ─────────────────────────────────────────────────────────────────────────────

def die(msg):
    print("\n[ERROR]", msg)
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
#  PART 1 — IMAGE → HEX
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  STEP 1 — Convert 2.png to sim_image.hex")
print("=" * 60)

if not os.path.exists(IMG_PATH):
    die(f"Cannot find '{IMG_PATH}' — make sure 2.png is in the same folder as this script (GUC/)")

# Open, convert to grayscale, resize to 28×28
img = Image.open(IMG_PATH).convert('L')
img = img.resize((28, 28), Image.LANCZOS)
pixels = np.array(img, dtype=np.float32)   # shape (28,28), values 0..255

# MNIST normalization + Q6 quantize
norm   = (pixels / 255.0 - MNIST_MEAN) / MNIST_STD
quant  = np.clip(np.round(norm * SCALE), -128, 127).astype(np.int8)
flat   = quant.flatten()   # 784 values, row-major

# Write hex file
os.makedirs("hw", exist_ok=True)
with open(HEX_OUT, 'w') as f:
    for v in flat:
        f.write(f"{int(v) & 0xFF:02X}\n")

print(f"  Written: {HEX_OUT}  ({len(flat)} bytes)")
print(f"  pixel[0]  = {flat[0]}   (background=-27, any other value = has signal)")
print(f"  pixel[783]= {flat[783]}")

if flat[0] == -27 and np.all(quant == -27):
    die("ALL pixels are background value -27 — image is completely blank or all white. Check 2.png.")

nonzero = np.sum(quant != -27)
print(f"  Non-background pixels: {nonzero}/784")
print("  [OK] Image has signal\n")

# ══════════════════════════════════════════════════════════════════════════════
#  PART 2 — LOAD WEIGHTS FROM HEX FILES
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  STEP 2 — Load weights from hex_weights/")
print("=" * 60)

def load_hex(fname, count):
    path = os.path.join(WEIGHTS_DIR, fname)
    if not os.path.exists(path):
        die(f"Missing weight file: {path}")
    vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                vals.append(int(line, 16))
    if len(vals) < count:
        die(f"{fname}: expected {count} entries, got {len(vals)}")
    # interpret as signed int8
    arr = np.array(vals[:count], dtype=np.uint8).view(np.int8)
    return arr.astype(np.int32)

c1w = load_hex("conv1_weights.hex", 150)      # 6 filters × 1ch × 5×5
c1b = load_hex("conv1_bias.hex",    6)
c2w = load_hex("conv2_weights.hex", 2400)     # 16 filters × 6ch × 5×5, order: (pos,ch)
c2b = load_hex("conv2_bias.hex",    16)
f1w = load_hex("fc1_weights.hex",   30720)    # 120×256
f1b = load_hex("fc1_bias.hex",      120)
f2w = load_hex("fc2_weights.hex",   10080)    # 84×120
f2b = load_hex("fc2_bias.hex",      84)
f3w = load_hex("fc3_weights.hex",   840)      # 10×84
f3b = load_hex("fc3_bias.hex",      10)

print(f"  conv1_w[0] = {c1w[0]}  (should be non-zero)")
print(f"  fc3_bias[6]= {f3b[6]}")
if c1w[0] == 0 and np.all(c1w == 0):
    die("All conv1 weights are zero — hex_weights/ folder has wrong files")
print("  [OK] Weights loaded\n")

# ══════════════════════════════════════════════════════════════════════════════
#  PART 3 — INTEGER REFERENCE INFERENCE
#  Matches RTL exactly: bias loaded as signed Q6, accumulate as int32,
#  ReLU+clip to int8, pool = max of 2×2 window.
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  STEP 3 — Integer reference inference")
print("=" * 60)

def relu_clip(x):
    """ReLU + saturate to int8 range [0,127]"""
    if x < 0:   return 0
    if x > 127*SCALE: return 127
    return x >> FRAC_BITS

def conv2d(inp, weights, biases, n_filt, in_h, in_w, in_ch, out_h, out_w):
    """
    inp     : flat int32 array [in_ch * in_h * in_w]
    weights : flat int32, RTL order = (pos, ch) for each filter
              i.e. weight[f][pos*in_ch + ch]
    returns : flat int32 array [n_filt * out_h * out_w]  (after ReLU+clip)
    """
    wt_per = 25 * in_ch   # weights per filter
    out = np.zeros(n_filt * out_h * out_w, dtype=np.int32)
    for f in range(n_filt):
        bias_acc = int(biases[f]) << FRAC_BITS
        for r in range(out_h):
            for c in range(out_w):
                acc = bias_acc
                mac_idx = 0
                for pos in range(25):
                    tr = pos // 5
                    tc = pos % 5
                    for ch in range(in_ch):
                        fv_idx = ch * in_h * in_w + (r+tr)*in_w + (c+tc)
                        w_idx  = f * wt_per + pos * in_ch + ch
                        acc   += int(inp[fv_idx]) * int(weights[w_idx])
                        mac_idx += 1
                out[f*out_h*out_w + r*out_w + c] = relu_clip(acc)
    return out

def maxpool2d(inp, n_filt, in_h, in_w):
    out_h = in_h // 2
    out_w = in_w // 2
    out = np.zeros(n_filt * out_h * out_w, dtype=np.int32)
    for f in range(n_filt):
        for r in range(out_h):
            for c in range(out_w):
                tl = inp[f*in_h*in_w + (2*r  )*in_w + (2*c  )]
                tr = inp[f*in_h*in_w + (2*r  )*in_w + (2*c+1)]
                bl = inp[f*in_h*in_w + (2*r+1)*in_w + (2*c  )]
                br = inp[f*in_h*in_w + (2*r+1)*in_w + (2*c+1)]
                out[f*out_h*out_w + r*out_w + c] = max(tl, tr, bl, br)
    return out

def fc_layer(inp, weights, biases, n_in, n_out, use_relu):
    out = np.zeros(n_out, dtype=np.int32)
    for n in range(n_out):
        acc = int(biases[n]) << FRAC_BITS
        for i in range(n_in):
            acc += int(inp[i]) * int(weights[n*n_in + i])
        if use_relu:
            out[n] = relu_clip(acc)
        else:
            # no ReLU for FC3: saturate to int8 signed
            v = acc >> FRAC_BITS
            out[n] = max(-128, min(127, v))
    return out

# ── Run inference ─────────────────────────────────────────────────────────────
img_flat = flat.astype(np.int32)   # 784

print("  Running CONV1 (6 filters, 28×28→24×24)...")
conv1 = conv2d(img_flat, c1w, c1b, 6, 28, 28, 1, 24, 24)

print("  Running POOL1 (24×24→12×12)...")
pool1 = maxpool2d(conv1, 6, 24, 24)

print("  Running CONV2 (16 filters, 12×12→8×8, 6ch)...")
conv2 = conv2d(pool1, c2w, c2b, 16, 12, 12, 6, 8, 8)

print("  Running POOL2 (8×8→4×4)...")
pool2 = maxpool2d(conv2, 16, 8, 8)

print("  Running FC1 (256→120)...")
fc1 = fc_layer(pool2, f1w, f1b, 256, 120, use_relu=True)

print("  Running FC2 (120→84)...")
fc2 = fc_layer(fc1, f2w, f2b, 120, 84, use_relu=True)

print("  Running FC3 (84→10)...")
fc3 = fc_layer(fc2, f3w, f3b, 84, 10, use_relu=False)

pred = int(np.argmax(fc3))

# ── Print results ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  RESULTS")
print("=" * 60)
print(f"  FC3 logits: {list(fc3)}")
print()
for i, v in enumerate(fc3):
    marker = " <── MAX" if i == pred else ""
    print(f"    class {i}: {v:4d}{marker}")
print()
print(f"  Predicted class : {pred}")
print(f"  Expected        : 2")
print(f"  {'PASS' if pred == 2 else 'FAIL — model predicts ' + str(pred) + ', not 2'}")
print()
print("=" * 60)
print("  COPY THESE VALUES AND SEND THEM:")
print(f"  pixel[0]   = {flat[0]}")
print(f"  FC3 logits = {list(fc3)}")
print(f"  Predicted  = {pred}")
print("=" * 60)
