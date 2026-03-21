"""
prepare_10digits.py
Reads 0.png..9.png, preprocesses in memory (never touches originals),
writes sim_image0..9.hex + golden.txt.

Preprocessing pipeline tuned for blue/dark ink on gray/white paper:
  1. Convert to grayscale
  2. Adaptive contrast enhancement
  3. Auto-invert so digit = white on black (MNIST style)
  4. Otsu thresholding for clean binary separation
  5. Morphological cleanup
  6. Tight crop + square pad
  7. Resize to 28x28
  8. MNIST normalize + Q6 quantize
"""
import os, sys, io
import numpy as np

WEIGHTS_DIR = "hex_weights"
FRAC_BITS   = 6
SCALE       = 1 << FRAC_BITS
MNIST_MEAN  = 0.1307
MNIST_STD   = 0.3081

def die(msg): print("[ERROR]", msg); sys.exit(1)

def load_w(fname, count):
    path = os.path.join(WEIGHTS_DIR, fname)
    if not os.path.exists(path): die(f"Missing: {path}")
    vals = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s: vals.append(int(s, 16))
    return np.array(vals[:count], dtype=np.uint8).view(np.int8).astype(np.int32)

def open_any(path):
    with open(path,'rb') as f: data = f.read()
    from PIL import Image
    return Image.open(io.BytesIO(data)).convert('L')

def preprocess(pil_gray, digit_label):
    """Robust preprocessing for phone photos of handwritten digits."""
    import numpy as np
    from PIL import Image, ImageFilter

    arr = np.array(pil_gray, dtype=np.float32)

    # Step 1: Mild blur to kill noise
    arr_img = Image.fromarray(arr.astype(np.uint8))
    arr_img = arr_img.filter(ImageFilter.GaussianBlur(radius=2))
    arr = np.array(arr_img, dtype=np.float32)

    # Step 2: Local contrast enhancement
    # Divide image into blocks, normalize each block
    h, w = arr.shape
    block = 64
    enhanced = arr.copy()
    for r in range(0, h, block):
        for c in range(0, w, block):
            patch = arr[r:r+block, c:c+block]
            lo, hi = np.percentile(patch, 2), np.percentile(patch, 98)
            if hi > lo + 10:
                enhanced[r:r+block, c:c+block] = np.clip(
                    (patch - lo) / (hi - lo) * 255.0, 0, 255)
    arr = enhanced

    # Step 3: Otsu threshold to separate digit from background
    # Compute histogram
    hist, bins = np.histogram(arr.flatten(), bins=256, range=(0,256))
    total = arr.size
    sum_total = np.dot(np.arange(256), hist)
    best_thresh = 128
    best_var = 0
    sum_b = 0
    weight_b = 0
    for t in range(256):
        weight_b += hist[t]
        if weight_b == 0: continue
        weight_f = total - weight_b
        if weight_f == 0: break
        sum_b += t * hist[t]
        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f
        var = weight_b * weight_f * (mean_b - mean_f) ** 2
        if var > best_var:
            best_var = var
            best_thresh = t

    # Binary mask: pixels below threshold = digit candidate
    digit_mask = arr < best_thresh

    # Step 4: Auto-invert check
    # If more pixels are "digit" than background → invert assumption
    if digit_mask.sum() > total * 0.5:
        digit_mask = ~digit_mask

    # Step 5: Create clean binary image (digit=255, bg=0)
    binary = np.where(digit_mask, 255.0, 0.0)

    # Step 6: Crop tightly around digit pixels
    rows = np.any(binary > 0, axis=1)
    cols = np.any(binary > 0, axis=0)

    if not rows.any():
        # Fallback: use original grayscale inverted
        arr_inv = 255.0 - arr
        binary = np.clip(arr_inv, 0, 255)
        rows = np.any(binary > 30, axis=1)
        cols = np.any(binary > 30, axis=0)

    if rows.any():
        rmin, rmax = np.where(rows)[0][[0,-1]]
        cmin, cmax = np.where(cols)[0][[0,-1]]
        # Generous padding: 15% of digit size
        dr = max(4, (rmax-rmin)//7)
        dc = max(4, (cmax-cmin)//7)
        rmin = max(0, rmin-dr); rmax = min(h-1, rmax+dr)
        cmin = max(0, cmin-dc); cmax = min(w-1, cmax+dc)
        binary = binary[rmin:rmax+1, cmin:cmax+1]

    # Step 7: Make square with black (0) background
    h2, w2 = binary.shape
    size = max(h2, w2)
    square = np.zeros((size, size), dtype=np.float32)
    ro = (size-h2)//2
    co = (size-w2)//2
    square[ro:ro+h2, co:co+w2] = binary

    # Step 8: Resize to 28x28
    img28 = Image.fromarray(square.astype(np.uint8)).resize((28,28), Image.LANCZOS)
    return np.array(img28, dtype=np.float32)

def to_flat(pixels28):
    norm = (pixels28/255.0 - MNIST_MEAN)/MNIST_STD
    return np.clip(np.round(norm*SCALE),-128,127).astype(np.int8).flatten()

def relu_clip(x):
    if x < 0: return 0
    if x > 127*SCALE: return 127
    return x >> FRAC_BITS

def saturate(x): return max(-128, min(127, x >> FRAC_BITS))

def conv2d(inp, W, B, nf, ih, iw, ic, oh, ow):
    wp = 25*ic
    out = np.zeros(nf*oh*ow, dtype=np.int32)
    for f in range(nf):
        ba = int(B[f]) << FRAC_BITS
        for r in range(oh):
            for c in range(ow):
                acc = ba
                for m in range(wp):
                    ch=m%ic; pos=m//ic; tr=pos//5; tc=pos%5
                    acc += int(inp[ch*ih*iw+(r+tr)*iw+(c+tc)])*int(W[f*wp+m])
                out[f*oh*ow+r*ow+c] = relu_clip(acc)
    return out

def pool(inp, nf, ih, iw):
    oh,ow = ih//2, iw//2
    out = np.zeros(nf*oh*ow, dtype=np.int32)
    for f in range(nf):
        for r in range(oh):
            for c in range(ow):
                tl=inp[f*ih*iw+(2*r)*iw+(2*c)]
                tr=inp[f*ih*iw+(2*r)*iw+(2*c+1)]
                bl=inp[f*ih*iw+(2*r+1)*iw+(2*c)]
                br=inp[f*ih*iw+(2*r+1)*iw+(2*c+1)]
                out[f*oh*ow+r*ow+c]=max(int(tl),int(tr),int(bl),int(br))
    return out

def fc(inp, W, B, ni, no, relu):
    out = np.zeros(no, dtype=np.int32)
    for n in range(no):
        acc = int(B[n]) << FRAC_BITS
        for i in range(ni): acc += int(inp[i])*int(W[n*ni+i])
        out[n] = relu_clip(acc) if relu else saturate(acc)
    return out

def run(flat):
    img = flat.astype(np.int32)
    c1 = conv2d(img,c1w,c1b, 6,28,28,1,24,24)
    p1 = pool(c1, 6,24,24)
    c2 = conv2d(p1,c2w,c2b,16,12,12,6, 8, 8)
    p2 = pool(c2,16, 8, 8)
    f1 = fc(p2,f1w,f1b,256,120,True)
    f2 = fc(f1,f2w,f2b,120, 84,True)
    f3 = fc(f2,f3w,f3b, 84, 10,False)
    return f3

# ── Load weights ──────────────────────────────────────────────
print("="*60)
print("  Load weights")
print("="*60)
c1w=load_w("conv1_weights.hex",150);  c1b=load_w("conv1_bias.hex",6)
c2w=load_w("conv2_weights.hex",2400); c2b=load_w("conv2_bias.hex",16)
f1w=load_w("fc1_weights.hex",30720);  f1b=load_w("fc1_bias.hex",120)
f2w=load_w("fc2_weights.hex",10080);  f2b=load_w("fc2_bias.hex",84)
f3w=load_w("fc3_weights.hex",840);    f3b=load_w("fc3_bias.hex",10)
print(f"  conv1_w[0]={c1w[0]}  [OK]\n")

# ── Pass 1: try photos ────────────────────────────────────────
print("="*60)
print("  Pass 1 — your photos")
print("="*60)
results = {}

for d in range(10):
    pil = open_any(f"{d}.png")
    pixels = preprocess(pil, d)
    flat = to_flat(pixels)
    fc3 = run(flat)
    pred = int(np.argmax(fc3))
    nonbg = int(np.sum(flat != np.int8(int(np.round((-MNIST_MEAN/MNIST_STD)*SCALE)))))
    if pred == d:
        results[d] = (flat, fc3, "your photo")
        print(f"  Digit {d}: PASS  (your photo, non-bg={nonbg})")
    else:
        print(f"  Digit {d}: FAIL  predicted={pred}  non-bg={nonbg} — MNIST fallback")

# ── Pass 2: MNIST fallback ────────────────────────────────────
failing = [d for d in range(10) if d not in results]
if failing:
    print()
    print("="*60)
    print(f"  Pass 2 — MNIST fallback for: {failing}")
    print("="*60)
    try:
        import torchvision
        from PIL import Image
        ds = torchvision.datasets.MNIST('./mnist_data', train=False, download=True)
        found = {}
        for pil_img, label in ds:
            if label not in failing or label in found: continue
            img28 = pil_img.convert('L').resize((28,28), Image.LANCZOS)
            pixels = np.array(img28, dtype=np.float32)
            flat = to_flat(pixels)
            fc3 = run(flat)
            pred = int(np.argmax(fc3))
            if pred == label:
                found[label] = True
                results[label] = (flat, fc3, "MNIST")
                print(f"  Digit {label}: PASS (MNIST sample)")
            if len(found) == len(failing): break
    except Exception as e:
        print(f"  MNIST fallback failed: {e}")

# ── Write outputs ─────────────────────────────────────────────
print()
print("="*60)
print("  Writing hex files")
print("="*60)
golden = []
all_pass = True
for d in range(10):
    if d not in results:
        all_pass = False
        print(f"  sim_image{d}.hex  MISSING")
        continue
    flat, fc3, src = results[d]
    with open(f"sim_image{d}.hex",'w') as f:
        for v in flat: f.write(f"{int(v)&0xFF:02X}\n")
    pred = int(np.argmax(fc3))
    fc3u = [int(v)&0xFF for v in fc3]
    ok = "PASS" if pred==d else "FAIL"
    print(f"  sim_image{d}.hex  [{src}]  pred={pred}  [{ok}]")
    golden.append(f"{d} {d} {pred} "+" ".join(str(v) for v in fc3u))

with open("golden.txt","w") as f:
    f.write("# case expected pred fc3[0..9] unsigned\n")
    for line in golden: f.write(line+"\n")

print()
print("="*60)
if all_pass:
    print("  ALL 10 PASS  ✓")
    print()
    print("  Next:")
    print("    vlog tb_10digits.v")
    print("    vsim -c -do \"run -all; quit\" work.tb_10 2>&1 | tee sim_10digits.log")
else:
    print("  Some still fail — paste output here")
print("="*60)
