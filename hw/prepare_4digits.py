"""
prepare_4digits.py
──────────────────
Converts 1.png, 2.png, 4.png, 6.png → sim_image0..3.hex
Runs integer reference inference on each.
Writes golden.txt for cross-checking against RTL simulation.

Run from hw/ folder:
    python prepare_4digits.py

Requirements: pip install pillow numpy
"""

import os, sys
import numpy as np
from PIL import Image

# ── YOUR 4 IMAGES ─────────────────────────────────────────────────────────────
IMAGES = [
    {"file": "1.png", "expected": 1, "hex_out": "sim_image0.hex"},
    {"file": "2.png", "expected": 2, "hex_out": "sim_image1.hex"},
    {"file": "3.png", "expected": 3, "hex_out": "sim_image2.hex"},
    {"file": "6.png", "expected": 6, "hex_out": "sim_image3.hex"},
]

WEIGHTS_DIR = "hex_weights"
FRAC_BITS   = 6
SCALE       = 1 << FRAC_BITS   # 64
MNIST_MEAN  = 0.1307
MNIST_STD   = 0.3081
# ─────────────────────────────────────────────────────────────────────────────

def die(msg):
    print("[ERROR]", msg); sys.exit(1)

def load_hex_weights(fname, count):
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
        die(f"{fname}: need {count} entries, got {len(vals)}")
    return np.array(vals[:count], dtype=np.uint8).view(np.int8).astype(np.int32)

def relu_clip(x):
    if x < 0:           return 0
    if x > 127 * SCALE: return 127
    return x >> FRAC_BITS

def saturate_signed(x):
    """FC3 only — no ReLU, clamp to signed int8."""
    v = x >> FRAC_BITS
    return max(-128, min(127, v))

def conv2d(inp, weights, biases, n_filt, in_h, in_w, in_ch, out_h, out_w):
    """
    RTL tap ordering (matches fixed conv_maxpool_block_final.v):
      tap_ch  = mac_idx % in_ch
      tap_pos = mac_idx // in_ch
      tr = tap_pos // 5,  tc = tap_pos % 5
    """
    wt_per = 25 * in_ch
    out = np.zeros(n_filt * out_h * out_w, dtype=np.int32)
    for f in range(n_filt):
        bias_acc = int(biases[f]) << FRAC_BITS
        for r in range(out_h):
            for c in range(out_w):
                acc = bias_acc
                for mac_idx in range(wt_per):
                    ch  = mac_idx % in_ch
                    pos = mac_idx // in_ch
                    tr  = pos // 5
                    tc  = pos % 5
                    fv_idx = ch * in_h * in_w + (r + tr) * in_w + (c + tc)
                    w_idx  = f * wt_per + mac_idx
                    acc   += int(inp[fv_idx]) * int(weights[w_idx])
                out[f * out_h * out_w + r * out_w + c] = relu_clip(acc)
    return out

def maxpool2d(inp, n_filt, in_h, in_w):
    oh, ow = in_h // 2, in_w // 2
    out = np.zeros(n_filt * oh * ow, dtype=np.int32)
    for f in range(n_filt):
        for r in range(oh):
            for c in range(ow):
                tl = inp[f*in_h*in_w + (2*r  )*in_w + (2*c  )]
                tr = inp[f*in_h*in_w + (2*r  )*in_w + (2*c+1)]
                bl = inp[f*in_h*in_w + (2*r+1)*in_w + (2*c  )]
                br = inp[f*in_h*in_w + (2*r+1)*in_w + (2*c+1)]
                out[f*oh*ow + r*ow + c] = max(int(tl), int(tr), int(bl), int(br))
    return out

def fc_layer(inp, weights, biases, n_in, n_out, use_relu):
    out = np.zeros(n_out, dtype=np.int32)
    for n in range(n_out):
        acc = int(biases[n]) << FRAC_BITS
        for i in range(n_in):
            acc += int(inp[i]) * int(weights[n * n_in + i])
        out[n] = relu_clip(acc) if use_relu else saturate_signed(acc)
    return out

def image_to_hex(img_path, hex_out):
    """Load PNG, MNIST-normalize, Q6-quantize, write hex. Returns flat int8."""
    img = Image.open(img_path).convert('L')
    img = img.resize((28, 28), Image.LANCZOS)
    pixels = np.array(img, dtype=np.float32)
    norm  = (pixels / 255.0 - MNIST_MEAN) / MNIST_STD
    quant = np.clip(np.round(norm * SCALE), -128, 127).astype(np.int8)
    flat  = quant.flatten()
    with open(hex_out, 'w') as f:
        for v in flat:
            f.write(f"{int(v) & 0xFF:02X}\n")
    return flat

def run_inference(flat):
    img_flat = flat.astype(np.int32)
    conv1 = conv2d(img_flat, c1w, c1b,  6, 28, 28, 1, 24, 24)
    pool1 = maxpool2d(conv1,  6, 24, 24)
    conv2 = conv2d(pool1,    c2w, c2b, 16, 12, 12, 6,  8,  8)
    pool2 = maxpool2d(conv2, 16,  8,  8)
    fc1   = fc_layer(pool2, f1w, f1b, 256, 120, use_relu=True)
    fc2   = fc_layer(fc1,   f2w, f2b, 120,  84, use_relu=True)
    fc3   = fc_layer(fc2,   f3w, f3b,  84,  10, use_relu=False)
    return fc3

# ── Load weights once ─────────────────────────────────────────────────────────
print("=" * 65)
print("  STEP 1 — Load weights")
print("=" * 65)
c1w = load_hex_weights("conv1_weights.hex", 150)
c1b = load_hex_weights("conv1_bias.hex",    6)
c2w = load_hex_weights("conv2_weights.hex", 2400)
c2b = load_hex_weights("conv2_bias.hex",    16)
f1w = load_hex_weights("fc1_weights.hex",   30720)
f1b = load_hex_weights("fc1_bias.hex",      120)
f2w = load_hex_weights("fc2_weights.hex",   10080)
f2b = load_hex_weights("fc2_bias.hex",      84)
f3w = load_hex_weights("fc3_weights.hex",   840)
f3b = load_hex_weights("fc3_bias.hex",      10)
print(f"  conv1_w[0]={c1w[0]}  fc3_bias[6]={f3b[6]}")
if c1w[0] == 0 and np.all(c1w == 0):
    die("All conv1 weights zero — wrong hex_weights/ folder")
print("  [OK] Weights loaded")
print()

# ── Process each image ────────────────────────────────────────────────────────
print("=" * 65)
print("  STEP 2 — Convert images + run reference inference")
print("=" * 65)

golden_lines = []
all_pass = True

for case_idx, cfg in enumerate(IMAGES):
    img_path = cfg["file"]
    expected = cfg["expected"]
    hex_out  = cfg["hex_out"]

    if not os.path.exists(img_path):
        die(f"Image not found: '{img_path}' — make sure it is in the hw/ folder")

    flat   = image_to_hex(img_path, hex_out)
    pixel0 = int(flat[0])

    if pixel0 == -27 and np.all(flat == -27):
        die(f"{img_path}: ALL pixels are background — image is blank or pure white")

    fc3    = run_inference(flat)
    pred   = int(np.argmax(fc3))
    fc3_u  = [int(v) & 0xFF for v in fc3]   # unsigned for BRAM comparison
    ok     = "PASS" if pred == expected else "FAIL"
    if pred != expected:
        all_pass = False

    print(f"  Case {case_idx}: {img_path}")
    print(f"    pixel[0]  = {pixel0}")
    print(f"    predicted = {pred}   expected = {expected}   [{ok}]")
    print(f"    FC3 signed  : {list(fc3)}")
    print(f"    FC3 unsigned: {fc3_u}   <- compare to RTL BRAM [6588..6597]")
    print()

    golden_lines.append(f"{case_idx} {expected} {pred} " + " ".join(str(v) for v in fc3_u))

# ── Write golden.txt ─────────────────────────────────────────────────────────
with open("golden.txt", "w") as f:
    f.write("# col0=case  col1=expected  col2=python_pred  col3..12=FC3[0..9] unsigned\n")
    for line in golden_lines:
        f.write(line + "\n")

print("=" * 65)
print("  STEP 3 — Summary")
print("=" * 65)
print(f"  Hex files written:")
for cfg in IMAGES:
    print(f"    {cfg['hex_out']}  ←  {cfg['file']}  (expected class {cfg['expected']})")
print()
print(f"  Golden reference written: golden.txt")
print()
if all_pass:
    print("  ALL 4 CASES PASS in Python integer reference  ✓")
else:
    print("  WARNING: One or more cases FAIL in Python reference.")
    print("  This means the model may not recognize that image.")
    print("  Use cleaner black-on-white MNIST-style digits.")
print()
print("  NEXT STEP — simulate:")
print("    (make sure sim_image0..3.hex are in hw/)")
print("    vlog tb_4cases.v")
print("    vsim -c -do \"run -all; quit\" work.tb")
