#!/usr/bin/env python3
"""
Bit-aligned reference for LeNet RTL (conv_maxpool_block_final + mac_unit FC).
Loads sim_image.hex + hex_weights/*.hex — no PyTorch required for int path.

Compare to simulation: argmax over FC3 act_out (same ReLU clip as mac_unit).
"""
from __future__ import annotations

import os
import sys

DATA_W = 8
ACC_W = 24
FRAC_BITS = 6
FILT = 5


def s24(x: int) -> int:
    x &= (1 << ACC_W) - 1
    if x & (1 << (ACC_W - 1)):
        x -= 1 << ACC_W
    return x


def sext8(b: int) -> int:
    b &= 0xFF
    if b & 0x80:
        b -= 0x100
    return b


def sext16(p: int) -> int:
    p &= 0xFFFF
    if p & 0x8000:
        p -= 0x10000
    return p


def add24(a: int, b: int) -> int:
    return s24(a + b)


def relu_clip_conv(acc: int) -> int:
    """Matches conv_maxpool_block_final.v relu_clip (unsigned 8b output)."""
    acc = s24(acc)
    if acc < 0:
        return 0
    # |x[ACC_W-2 : DATA_W+FRAC_BITS-1]|  -> bits 22 down to 13
    hi = ACC_W - 2
    lo = DATA_W + FRAC_BITS - 1
    ovf = (acc >> lo) & ((1 << (hi - lo + 1)) - 1)
    if ovf != 0:
        return 0x7F
    # x[DATA_W+FRAC_BITS-2 : FRAC_BITS] = [12:6]
    return (acc >> FRAC_BITS) & 0x7F


def act_out_mac(acc: int) -> int:
    """Matches mac_unit_final.v act_out (signed 8b, ReLU + overflow clip)."""
    acc = s24(acc)
    if acc < 0:
        return 0
    hi = ACC_W - 2
    lo = DATA_W + FRAC_BITS - 1
    ovf = (acc >> lo) & ((1 << (hi - lo + 1)) - 1)
    if ovf != 0:
        return 0x7F
    v = (acc >> FRAC_BITS) & 0x7F
    return v if v < 0x80 else v - 0x100  # int8-ish for display


def read_memh(path: str) -> list[int]:
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            out.append(int(line, 16) & 0xFF)
    return out


def load_image(path: str) -> list[int]:
    vals = read_memh(path)
    if len(vals) != 784:
        raise ValueError(f"{path}: expected 784 bytes, got {len(vals)}")
    return vals


def conv_mac_accum(bias: int, weights: list[int], inputs: list[int]) -> int:
    acc = s24(sext8(bias) << FRAC_BITS)
    for w, x in zip(weights, inputs):
        p = sext8(w) * sext8(x)
        acc = add24(acc, sext16(p))
    return acc


def rtl_tap(mac_idx: int, in_ch: int) -> tuple[int, int, int]:
    """
    Match conv_maxpool_block_final.v (fixed: full tap_pos for 5x5 indexing).
    """
    if in_ch == 1:
        tap_ch = 0
        tap_pos = mac_idx
    else:
        tap_ch = mac_idx % 6
        tap_pos = mac_idx // 6
    tap_row = tap_pos // 5
    tap_col = tap_pos % 5
    return tap_ch, tap_row, tap_col


def maxpool2d(channels: list[list[list[int]]], h: int, w: int) -> list[list[list[int]]]:
    """channels[f][r][c] -> pool ph=f, pr, pc max over 2x2."""
    ph, pw = h // 2, w // 2
    out = []
    for f in range(len(channels)):
        plane = []
        for pr in range(ph):
            row = []
            for pc in range(pw):
                r0, c0 = 2 * pr, 2 * pc
                vals = [
                    channels[f][r0][c0],
                    channels[f][r0][c0 + 1],
                    channels[f][r0 + 1][c0],
                    channels[f][r0 + 1][c0 + 1],
                ]
                row.append(max(vals))
            plane.append(row)
        out.append(plane)
    return out


def run_lenet_int(img: list[int], hex_dir: str) -> tuple[list[int], dict]:
    # --- weights ---
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

    assert len(c1w) == 150 and len(c1b) == 6
    assert len(c2w) == 2400 and len(c2b) == 16
    assert len(f1w) == 30720 and len(f1b) == 120
    assert len(f2w) == 10080 and len(f2b) == 84
    assert len(f3w) == 840 and len(f3b) == 10

    # img as 28x28 uint8 (RTL: BRAM stores UART bytes)
    H0, W0 = 28, 28
    inp = [img[r * W0 + c] for r in range(H0) for c in range(W0)]

    # --- CONV1: 6 filters, 1 ch, 24x24 out ---
    n_f1, in_h, in_w, in_ch = 6, 28, 28, 1
    out_h, out_w = in_h - FILT + 1, in_w - FILT + 1  # 24
    wt_per1 = FILT * FILT * in_ch  # 25
    conv1 = []
    for filt in range(n_f1):
        plane = []
        for orow in range(out_h):
            row = []
            for ocol in range(out_w):
                wts = c1w[filt * wt_per1 : (filt + 1) * wt_per1]
                patches = []
                for mac_idx in range(wt_per1):
                    _, tr, tc = rtl_tap(mac_idx, in_ch)
                    r = orow + tr
                    c = ocol + tc
                    patches.append(inp[r * W0 + c])
                acc = conv_mac_accum(c1b[filt], wts, patches)
                row.append(relu_clip_conv(acc))
            plane.append(row)
        conv1.append(plane)

    pool1 = maxpool2d(conv1, out_h, out_w)  # 6 x 12 x 12

    # --- CONV2: 16 filters, 6 ch, 8x8 out ---
    n_f2, in_h2, in_w2, in_ch2 = 16, 12, 12, 6
    out_h2, out_w2 = in_h2 - FILT + 1, in_w2 - FILT + 1  # 8
    wt_per2 = FILT * FILT * in_ch2  # 150
    conv2 = []
    for filt in range(n_f2):
        plane = []
        for orow in range(out_h2):
            row = []
            for ocol in range(out_w2):
                wts = c2w[filt * wt_per2 : (filt + 1) * wt_per2]
                patches = []
                for mac_idx in range(wt_per2):
                    ch, tr, tc = rtl_tap(mac_idx, in_ch2)
                    r = orow + tr
                    c = ocol + tc
                    patches.append(pool1[ch][r][c])
                acc = conv_mac_accum(c2b[filt], wts, patches)
                row.append(relu_clip_conv(acc))
            plane.append(row)
        conv2.append(plane)

    pool2 = maxpool2d(conv2, out_h2, out_w2)  # 16 x 4 x 4 = 256

    # Flatten: filter-major, row-major (matches typical RTL linear addr)
    fv = []
    for f in range(16):
        for r in range(4):
            for c in range(4):
                fv.append(pool2[f][r][c])

    def fc_layer(feat: list[int], n_in: int, n_out: int, w_flat: list[int], b_flat: list[int], use_relu: bool):
        out = []
        W = [[w_flat[j * n_in + i] for i in range(n_in)] for j in range(n_out)]
        for j in range(n_out):
            acc = s24(sext8(b_flat[j]) << FRAC_BITS)
            for i in range(n_in):
                p = sext8(feat[i]) * sext8(W[j][i])
                acc = add24(acc, sext16(p))
            out.append(act_out_mac(acc) if use_relu else act_out_mac(acc))
        return out

    fc1 = fc_layer(fv, 256, 120, f1w, f1b, True)
    fc2 = fc_layer(fc1, 120, 84, f2w, f2b, True)
    fc3 = fc_layer(fc2, 84, 10, f3w, f3b, True)

    meta = {
        "pool2_first8": fv[:8],
        "fc3_logits_int8": fc3,
        "pred": max(range(10), key=lambda i: fc3[i]),
    }
    return fc3, meta


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(root, "sim_image.hex")
    hex_dir = os.path.join(root, "hex_weights")
    if not os.path.isfile(img_path):
        print("Missing sim_image.hex", file=sys.stderr)
        sys.exit(1)
    img = load_image(img_path)
    fc3, meta = run_lenet_int(img, hex_dir)

    print("=== Python int reference (matches RTL fixed-point rules) ===")
    print(f"pool2[0:8] = {meta['pool2_first8']}")
    print(f"FC3 act (int): {fc3}")
    print(f"Argmax pred: {meta['pred']}")

    # Optional: PyTorch same architecture on normalized tensor (like image_to_sim)
    try:
        import torch
        import torch.nn as nn
        import torchvision.transforms as T
        from PIL import Image
        import numpy as np

        class LeNet5(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 6, 5)
                self.pool1 = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.pool2 = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(256, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.pool1(self.relu(self.conv1(x)))
                x = self.pool2(self.relu(self.conv2(x)))
                x = x.view(x.size(0), -1)
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return self.fc3(x)

        pth = os.path.join(root, "lenet5_trained.pth")
        if os.path.isfile(pth):
            arr = np.array(img, dtype=np.uint8).reshape(28, 28)
            im = Image.fromarray(arr)
            transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
            x = transform(im).unsqueeze(0)
            m = LeNet5()
            m.load_state_dict(torch.load(pth, map_location="cpu"))
            m.eval()
            with torch.no_grad():
                logits = m(x).squeeze().tolist()
            sw_pred = max(range(10), key=lambda i: logits[i])
            print("\n=== PyTorch float (MNIST norm, same checkpoint) ===")
            print(f"logits: {[round(v, 4) for v in logits]}")
            print(f"argmax: {sw_pred}")
    except Exception as e:
        print(f"\n(PyTorch path skipped: {e})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
