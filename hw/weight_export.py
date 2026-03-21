# =============================================================
#  Weight Export Script — LeNet-5 → FPGA ROM Hex Files
#  Target: Basys3 (Artix-7 XC7A35T), Vivado 2019.1
#  Fixed-point: 8-bit signed (1 sign + 1 integer + 6 fractional)
# =============================================================

import torch
import torch.nn as nn
import numpy as np
import os

# ── Configuration ──────────────────────────────────────────
FRAC_BITS  = 6          # 6 fractional bits
SCALE      = 2**FRAC_BITS  # 64
BIT_WIDTH  = 8          # 8-bit signed total
MAX_VAL    =  (2**(BIT_WIDTH-1)) - 1   #  127
MIN_VAL    = -(2**(BIT_WIDTH-1))       # -128
OUTPUT_DIR = "hex_weights"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── LeNet-5 Model (must match training script exactly) ──────
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1  = nn.Conv2d(1,  6,  5, 1, 0)
        self.relu1  = nn.ReLU()
        self.pool1  = nn.MaxPool2d(2, 2)
        self.conv2  = nn.Conv2d(6,  16, 5, 1, 0)
        self.relu2  = nn.ReLU()
        self.pool2  = nn.MaxPool2d(2, 2)
        self.fc1    = nn.Linear(256, 120)
        self.relu3  = nn.ReLU()
        self.fc2    = nn.Linear(120, 84)
        self.relu4  = nn.ReLU()
        self.fc3    = nn.Linear(84,  10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        return self.fc3(x)

# ── Helper: float → 8-bit signed fixed-point ───────────────
def to_fixed(val):
    """Convert float to 8-bit signed fixed-point integer."""
    scaled = int(round(val * SCALE))
    return max(MIN_VAL, min(MAX_VAL, scaled))

def to_hex8(val):
    """Convert to 2's complement 8-bit hex string."""
    v = to_fixed(val)
    if v < 0:
        v = v + 256   # 2's complement
    return f"{v:02X}"

# ── Helper: write .hex file (Vivado $readmemh format) ───────
def write_hex(filename, data_flat, label):
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w') as f:
        for val in data_flat:
            f.write(to_hex8(val) + "\n")
    print(f"  {label:35s} -> {filename}  ({len(data_flat)} values)")
    return len(data_flat)

# ── Helper: write .coe file (Vivado Block Memory format) ────
def write_coe(filename, data_flat):
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w') as f:
        f.write("memory_initialization_radix=16;\n")
        f.write("memory_initialization_vector=\n")
        for i, val in enumerate(data_flat):
            sep = "," if i < len(data_flat)-1 else ";"
            f.write(to_hex8(val) + sep + "\n")

# ── Load trained model ───────────────────────────────────────
print("Loading trained model: lenet5_trained.pth")
model = LeNet5()
model.load_state_dict(torch.load("lenet5_trained.pth",
                                  map_location='cpu'))
model.eval()
print("Model loaded successfully.\n")

print("=" * 60)
print("Exporting weights as 8-bit signed fixed-point")
print(f"  FRAC_BITS = {FRAC_BITS}, SCALE = {SCALE}")
print(f"  Range: [{MIN_VAL}, {MAX_VAL}] -> float [{MIN_VAL/SCALE:.4f}, {MAX_VAL/SCALE:.4f}]")
print("=" * 60)

total_params = 0

# ── CONV1 weights: shape [6, 1, 5, 5] ────────────────────────
# Stored as: filter0_row0..row4, filter1_row0..row4, ...
# Each filter = 25 values, 6 filters total = 150 values
w = model.conv1.weight.detach().numpy()   # [6,1,5,5]
b = model.conv1.bias.detach().numpy()     # [6]
flat_w = w.reshape(-1).tolist()
flat_b = b.reshape(-1).tolist()
total_params += write_hex("conv1_weights.hex", flat_w,
                           "CONV1 weights [6×1×5×5=150]")
total_params += write_hex("conv1_bias.hex",    flat_b,
                           "CONV1 biases  [6]")
write_coe("conv1_weights.coe", flat_w)
write_coe("conv1_bias.coe",    flat_b)

# ── CONV2 weights: shape [16, 6, 5, 5] ───────────────────────
# RTL uses mac_idx = tap_pos*6 + tap_ch (position-major, then channel).
# Reorder from PyTorch (ch, pos) to match RTL (pos, ch).
w = model.conv2.weight.detach().numpy()   # [16,6,5,5]
b = model.conv2.bias.detach().numpy()     # [16]
flat_w = []
for f in range(16):
    for pos in range(25):
        kh, kw = pos // 5, pos % 5
        for c in range(6):
            flat_w.append(float(w[f, c, kh, kw]))
flat_b = b.reshape(-1).tolist()
total_params += write_hex("conv2_weights.hex", flat_w,
                           "CONV2 weights [16×6×5×5=2400]")
total_params += write_hex("conv2_bias.hex",    flat_b,
                           "CONV2 biases  [16]")
write_coe("conv2_weights.coe", flat_w)
write_coe("conv2_bias.coe",    flat_b)

# ── FC1 weights: shape [120, 256] ────────────────────────────
w = model.fc1.weight.detach().numpy()     # [120,256]
b = model.fc1.bias.detach().numpy()       # [120]
flat_w = w.reshape(-1).tolist()
flat_b = b.reshape(-1).tolist()
total_params += write_hex("fc1_weights.hex", flat_w,
                           "FC1 weights   [120×256=30720]")
total_params += write_hex("fc1_bias.hex",    flat_b,
                           "FC1 biases    [120]")
write_coe("fc1_weights.coe", flat_w)
write_coe("fc1_bias.coe",    flat_b)

# ── FC2 weights: shape [84, 120] ─────────────────────────────
w = model.fc2.weight.detach().numpy()     # [84,120]
b = model.fc2.bias.detach().numpy()       # [84]
flat_w = w.reshape(-1).tolist()
flat_b = b.reshape(-1).tolist()
total_params += write_hex("fc2_weights.hex", flat_w,
                           "FC2 weights   [84×120=10080]")
total_params += write_hex("fc2_bias.hex",    flat_b,
                           "FC2 biases    [84]")
write_coe("fc2_weights.coe", flat_w)
write_coe("fc2_bias.coe",    flat_b)

# ── FC3 weights: shape [10, 84] ──────────────────────────────
w = model.fc3.weight.detach().numpy()     # [10,84]
b = model.fc3.bias.detach().numpy()       # [10]
flat_w = w.reshape(-1).tolist()
flat_b = b.reshape(-1).tolist()
total_params += write_hex("fc3_weights.hex", flat_w,
                           "FC3 weights   [10×84=840]")
total_params += write_hex("fc3_bias.hex",    flat_b,
                           "FC3 biases    [10]")
write_coe("fc3_weights.coe", flat_w)
write_coe("fc3_bias.coe",    flat_b)

print("=" * 60)
print(f"Total values exported: {total_params:,}")
print(f"Total memory:          {total_params:,} bytes = {total_params/1024:.1f} KB")
print(f"Output directory:      ./{OUTPUT_DIR}/")
print("=" * 60)

# ── Verification: check accuracy loss from quantization ─────
print("\nQuantization accuracy check...")
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader

transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,),(0.3081,))])
test_ds  = torchvision.datasets.MNIST('./data', train=False,
                                       download=True, transform=transform)
test_ld  = DataLoader(test_ds, batch_size=256, shuffle=False)

# Quantize model weights in-place
with torch.no_grad():
    for name, param in model.named_parameters():
        quantized = torch.tensor(
            [to_fixed(v.item()) / SCALE for v in param.flatten()],
            dtype=torch.float32
        ).reshape(param.shape)
        param.copy_(quantized)

model.eval()
correct = total = 0
with torch.no_grad():
    for imgs, labels in test_ld:
        out = model(imgs)
        pred = out.argmax(1)
        correct += (pred == labels).sum().item()
        total   += labels.size(0)

print(f"  Accuracy after 8-bit quantization: {100*correct/total:.2f}%")
print("  (should be close to original 99.25%)")
print("\nAll hex files ready for Vivado ROM initialization!")
