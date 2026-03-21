"""Trace layer outputs: quantized float vs int ref to find divergence."""
import os
import numpy as np

# Shared helpers
FRAC_BITS = 6
SCALE = 2 ** 6
ACC_W, DATA_W = 24, 8

def s24(x):
    x = int(x) & 0xFFFFFF
    if x & 0x800000:
        x -= 0x1000000
    return x

def sext8(b):
    b &= 0xFF
    if b >= 128:
        b -= 256
    return b

def sext16(p):
    p &= 0xFFFF
    if p >= 0x8000:
        p -= 0x10000
    return p

def add24(a, b):
    return s24(a + b)

def relu_clip_conv(acc):
    acc = s24(acc)
    if acc < 0:
        return 0
    if (acc >> 13) & 0x3FF:
        return 0x7F
    return (acc >> 6) & 0x7F

def act_out_mac(acc):
    acc = s24(acc)
    if acc < 0:
        return 0
    if (acc >> 13) & 0x3FF:
        return 0x7F
    return (acc >> 6) & 0x7F

def read_memh(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                out.append(int(line, 16) & 0xFF)
    return out

root = os.path.dirname(os.path.abspath(__file__))
hex_dir = os.path.join(root, "hex_weights")
img = read_memh(os.path.join(root, "sim_image.hex"))
c1w = read_memh(os.path.join(hex_dir, "conv1_weights.hex"))
c1b = read_memh(os.path.join(hex_dir, "conv1_bias.hex"))
c2w = read_memh(os.path.join(hex_dir, "conv2_weights.hex"))
c2b = read_memh(os.path.join(hex_dir, "conv2_bias.hex"))
f1w = read_memh(os.path.join(hex_dir, "fc1_weights.hex"))
f1b = read_memh(os.path.join(hex_dir, "fc1_bias.hex"))
f2w = read_memh(os.path.join(hex_dir, "fc2_weights.hex"))
f2b = read_memh(os.path.join(hex_dir, "fc2_bias.hex"))
f3w = read_memh(os.path.join(hex_dir, "fc3_weights.hex"))
f3b = read_memh(os.path.join(hex_dir, "fc3_bias.hex"))

def conv_mac(bias, wts, inp):
    acc = s24(sext8(bias) << FRAC_BITS)
    for w, x in zip(wts, inp):
        acc = add24(acc, sext16(sext8(w) * sext8(x)))
    return acc

FILT = 5

# CONV1
inp = [img[r*28+c] for r in range(28) for c in range(28)]
conv1_out = []
for filt in range(6):
    for orow in range(24):
        for ocol in range(24):
            wts = c1w[filt*25:(filt+1)*25]
            patches = []
            for mi in range(25):
                tr, tc = mi // 5, mi % 5
                patches.append(inp[(orow+tr)*28 + (ocol+tc)])
            acc = conv_mac(c1b[filt], wts, patches)
            conv1_out.append(relu_clip_conv(acc))

# Pool1: 6x12x12
conv1_3d = [conv1_out[f*576:(f+1)*576] for f in range(6)]
pool1 = []
for f in range(6):
    for pr in range(12):
        for pc in range(12):
            r0, c0 = 2*pr, 2*pc
            v = max(conv1_3d[f][r0*24+c0], conv1_3d[f][r0*24+c0+1],
                    conv1_3d[f][(r0+1)*24+c0], conv1_3d[f][(r0+1)*24+c0+1])
            pool1.append(v)

print("CONV1 first 4:", conv1_out[:4])
print("POOL1 first 8:", pool1[:8])

# CONV2
wt_per2 = 150
conv2_out = []
for filt in range(16):
    for orow in range(8):
        for ocol in range(8):
            wts = c2w[filt*wt_per2:(filt+1)*wt_per2]
            patches = []
            for mi in range(150):
                ch = mi % 6
                pos = mi // 6
                tr, tc = pos // 5, pos % 5
                r, c = orow + tr, ocol + tc
                idx = ch * 144 + r * 12 + c
                patches.append(pool1[idx])
            acc = conv_mac(c2b[filt], wts, patches)
            conv2_out.append(relu_clip_conv(acc))

pool2 = []
for f in range(16):
    for pr in range(4):
        for pc in range(4):
            r0, c0 = 2*pr, 2*pc
            base = f * 64
            v = max(conv2_out[base + r0*8+c0], conv2_out[base + r0*8+c0+1],
                    conv2_out[base + (r0+1)*8+c0], conv2_out[base + (r0+1)*8+c0+1])
            pool2.append(v)

print("POOL2 first 8:", pool2[:8])

# FC
def fc(feat, n_in, n_out, w, b):
    out = []
    for j in range(n_out):
        acc = s24(sext8(b[j]) << FRAC_BITS)
        for i in range(n_in):
            p = sext8(feat[i]) * sext8(w[j*n_in + i])
            acc = add24(acc, sext16(p))
        out.append(act_out_mac(acc))
    return out

fc1_out = fc(pool2, 256, 120, f1w, f1b)
fc2_out = fc(fc1_out, 120, 84, f2w, f2b)
fc3_out = fc(fc2_out, 84, 10, f3w, f3b)

print("FC1 first 4:", fc1_out[:4])
print("FC2 first 4:", fc2_out[:4])
print("FC3:", fc3_out)
print("Argmax:", max(range(10), key=lambda i: fc3_out[i]))
