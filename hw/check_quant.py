"""Quick check: does quantized (float) model predict 6 on our image?"""
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

FRAC_BITS = 6
SCALE = 2 ** FRAC_BITS

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

def to_fixed(val):
    scaled = int(round(val * SCALE))
    return max(-128, min(127, scaled)) / SCALE

root = os.path.dirname(os.path.abspath(__file__))
# Load hex as normalized+quantized (signed)
with open(os.path.join(root, "sim_image.hex")) as f:
    raw = [int(l.strip(), 16) for l in f if l.strip()]
pixels = [(b - 256 if b >= 128 else b) for b in raw]  # twos complement
# Dequantize to float for PyTorch
x_float = np.array(pixels, dtype=np.float64) / SCALE
x_float = x_float.reshape(1, 1, 28, 28)
t = torch.from_numpy(x_float).float()

model = LeNet5()
model.load_state_dict(torch.load(os.path.join(root, "lenet5_trained.pth"), map_location="cpu"))

# Quantize weights in-place (same as weight_export)
with torch.no_grad():
    for name, param in model.named_parameters():
        q = torch.tensor([to_fixed(v.item()) for v in param.flatten()], dtype=torch.float32).reshape(param.shape)
        param.copy_(q)

# Hook to capture intermediates
conv1_o = pool1_o = conv2_o = pool2_o = fc1_o = fc2_o = None

def h1(m, i, o): global conv1_o; conv1_o = o.detach()
def h2(m, i, o): global pool1_o; pool1_o = o.detach()
def h3(m, i, o): global conv2_o; conv2_o = o.detach()
def h4(m, i, o): global pool2_o; pool2_o = o.detach()
def h5(m, i, o): global fc1_o; fc1_o = o.detach()
def h6(m, i, o): global fc2_o; fc2_o = o.detach()

model.conv1.register_forward_hook(h1)
model.pool1.register_forward_hook(h2)
model.conv2.register_forward_hook(h3)
model.pool2.register_forward_hook(h4)
model.fc1.register_forward_hook(h5)
model.fc2.register_forward_hook(h6)

model.eval()
with torch.no_grad():
    out = model(t)

logits = out.squeeze().tolist()
pred = max(range(10), key=lambda i: logits[i])
print("Quantized float model on FPGA input (norm+Q6):")
# pool2 shape [1,16,4,4] - flatten f,h,w to match int order
p2 = [round(float(pool2_o[0,f,h,w].item()), 2) for f in range(16) for h in range(4) for w in range(4)]
print("  pool2 first 8:", p2[:8])
print("  fc3 logits:", [round(v, 3) for v in logits])
print("  pred:", pred)
