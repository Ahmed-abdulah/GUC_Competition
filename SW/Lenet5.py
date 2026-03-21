# =============================================================
#  LeNet-5 CNN — Exact Software Model (as per paper)
#  Reference: Mukhopadhyay et al., Computers & Electrical
#             Engineering 97 (2022) 107628
#
#  Dataset  : MNIST
#  Framework: PyTorch
# =============================================================

# ─────────────────────────────────────────────
# STEP 0 — Install dependencies (run once)
# pip install torch torchvision matplotlib numpy
# ─────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────
# STEP 1 — Configuration (all hyperparameters)
# ─────────────────────────────────────────────

CONFIG = {
    "batch_size"   : 64,        # Number of images per training batch
    "epochs"       : 20,        # Number of full passes over the dataset
    "learning_rate": 0.001,     # Adam optimizer learning rate
    "num_classes"  : 10,        # Digits 0–9
    "seed"         : 42,        # For reproducibility
    "save_path"    : "lenet5_trained.pth",  # Where to save trained weights
}

# Fix random seed for reproducibility
torch.manual_seed(CONFIG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────────
# STEP 2 — Data Loading & Preprocessing
# ─────────────────────────────────────────────
#
# MNIST images are 28×28 pixels, grayscale.
# The paper's LeNet-5 uses 28×28 input directly
# (unlike the original 1998 version which used 32×32).
#
# Transforms applied:
#   ToTensor()     → converts PIL image to tensor [0.0, 1.0]
#   Normalize()    → zero-mean, unit-variance normalization
#                    mean=0.1307, std=0.3081 are the
#                    standard MNIST dataset statistics

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load training set (60,000 images)
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Download and load test set (10,000 images)
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True       # Shuffle training data every epoch
)

test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False      # No need to shuffle test data
)

print(f"Training samples : {len(train_dataset)}")
print(f"Test samples     : {len(test_dataset)}")


# ─────────────────────────────────────────────
# STEP 3 — LeNet-5 Model Definition
# ─────────────────────────────────────────────
#
# Exact architecture matching the reference paper:
#
#  Input       : 28×28×1
#  CONV1       : 6 filters, 5×5  → 24×24×6
#  MAXPOOL1    : 2×2             → 12×12×6
#  CONV2       : 16 filters, 5×5 → 8×8×16
#  MAXPOOL2    : 2×2             → 4×4×16
#  FLATTEN     : 4×4×16 = 256
#  FC1 (HL1)   : 120 neurons
#  FC2 (HL2)   : 84 neurons
#  OUTPUT (OL) : 10 neurons
#
# Activation : ReLU (as stated in paper Section 2.1)
# Loss       : Cross-Entropy (as stated in paper Section 2.3)

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        # ── Convolutional Block 1 ──────────────────
        # Input : 1 channel (grayscale), 28×28
        # Output: 6 feature maps, 24×24
        # Kernel: 5×5, no padding, stride=1
        # Formula: output_size = (28 - 5) / 1 + 1 = 24
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding=0
        )
        self.relu1 = nn.ReLU()

        # MaxPool1: 2×2 window, stride=2
        # Output: 6 feature maps, 12×12
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Convolutional Block 2 ──────────────────
        # Input : 6 channels, 12×12
        # Output: 16 feature maps, 8×8
        # Kernel: 5×5, no padding, stride=1
        # Formula: output_size = (12 - 5) / 1 + 1 = 8
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0
        )
        self.relu2 = nn.ReLU()

        # MaxPool2: 2×2 window, stride=2
        # Output: 16 feature maps, 4×4
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Fully Connected Layers ─────────────────
        # Flatten: 16 × 4 × 4 = 256 inputs

        # FC1 (Hidden Layer 1): 256 → 120
        self.fc1  = nn.Linear(16 * 4 * 4, 120)
        self.relu3 = nn.ReLU()

        # FC2 (Hidden Layer 2): 120 → 84
        self.fc2  = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()

        # Output Layer: 84 → 10 (one per digit class)
        # Note: No softmax here — nn.CrossEntropyLoss
        # applies softmax internally
        self.fc3  = nn.Linear(84, num_classes)

    def forward(self, x):
        # x shape: [batch, 1, 28, 28]

        # CONV1 → ReLU → MAXPOOL1
        x = self.pool1(self.relu1(self.conv1(x)))
        # x shape: [batch, 6, 12, 12]

        # CONV2 → ReLU → MAXPOOL2
        x = self.pool2(self.relu2(self.conv2(x)))
        # x shape: [batch, 16, 4, 4]

        # Flatten
        x = x.view(x.size(0), -1)
        # x shape: [batch, 256]

        # FC1 → ReLU
        x = self.relu3(self.fc1(x))
        # x shape: [batch, 120]

        # FC2 → ReLU
        x = self.relu4(self.fc2(x))
        # x shape: [batch, 84]

        # Output layer (logits)
        x = self.fc3(x)
        # x shape: [batch, 10]

        return x


# Instantiate model and move to device (GPU if available)
model = LeNet5(num_classes=CONFIG["num_classes"]).to(device)
print("\nModel Architecture:")
print(model)

# Count total trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal trainable parameters: {total_params:,}")


# ─────────────────────────────────────────────
# STEP 4 — Loss Function & Optimizer
# ─────────────────────────────────────────────
#
# Cross-Entropy Loss: as specified in paper Eq.(5)
#   Loss = -Σ y_true * log(y_pred)
#
# Adam Optimizer: adaptive learning rate,
#   works better than plain SGD for this task

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=CONFIG["learning_rate"]
)

# Learning rate scheduler: reduces LR by 50%
# every 5 epochs if no improvement
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.5
)


# ─────────────────────────────────────────────
# STEP 5 — Training Loop
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one full epoch, return average loss and accuracy."""
    model.train()
    total_loss     = 0.0
    correct        = 0
    total_samples  = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # ① Forward pass
        outputs = model(images)

        # ② Compute loss
        loss = criterion(outputs, labels)

        # ③ Zero gradients from previous step
        optimizer.zero_grad()

        # ④ Backpropagation
        loss.backward()

        # ⑤ Update weights
        optimizer.step()

        # Track metrics
        total_loss    += loss.item() * images.size(0)
        _, predicted   = torch.max(outputs, 1)
        correct       += (predicted == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct / total_samples
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate model on a dataset, return average loss and accuracy."""
    model.eval()
    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    with torch.no_grad():  # No gradient computation needed
        for images, labels in loader:
            images  = images.to(device)
            labels  = labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss    += loss.item() * images.size(0)
            _, predicted   = torch.max(outputs, 1)
            correct       += (predicted == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct / total_samples
    return avg_loss, accuracy


# ─────────────────────────────────────────────
# STEP 6 — Run Training
# ─────────────────────────────────────────────

train_losses, train_accs = [], []
test_losses,  test_accs  = [], []
best_accuracy = 0.0

print("\n" + "="*60)
print("Starting Training...")
print("="*60)

for epoch in range(1, CONFIG["epochs"] + 1):

    # Train
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )

    # Evaluate
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device
    )

    # Step the learning rate scheduler
    scheduler.step()

    # Save history for plotting
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    # Save best model
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        torch.save(model.state_dict(), CONFIG["save_path"])
        saved_marker = " ← Best model saved"
    else:
        saved_marker = ""

    print(
        f"Epoch [{epoch:02d}/{CONFIG['epochs']}] | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
        f"Test Loss: {test_loss:.4f}  | Test Acc: {test_acc:.2f}%"
        f"{saved_marker}"
    )

print("="*60)
print(f"Training complete. Best Test Accuracy: {best_accuracy:.2f}%")
print("="*60)


# ─────────────────────────────────────────────
# STEP 7 — Plot Training Curves
# ─────────────────────────────────────────────

epochs_range = range(1, CONFIG["epochs"] + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss plot
axes[0].plot(epochs_range, train_losses, label="Train Loss", color="blue")
axes[0].plot(epochs_range, test_losses,  label="Test Loss",  color="red")
axes[0].set_title("Loss vs Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

# Accuracy plot
axes[1].plot(epochs_range, train_accs, label="Train Accuracy", color="blue")
axes[1].plot(epochs_range, test_accs,  label="Test Accuracy",  color="red")
axes[1].set_title("Accuracy vs Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
axes[1].grid(True)

plt.suptitle("LeNet-5 Training on MNIST", fontsize=14)
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()
print("Training curves saved to training_curves.png")


# ─────────────────────────────────────────────
# STEP 8 — Export Weights for FPGA
# ─────────────────────────────────────────────
#
# After training, extract weights as fixed-point
# integers (7-bit as used in the paper) for
# loading into the FPGA ROM/RAM.
#
# FRACTIONAL BITS = 6  (matches paper's Fig. 3
# which shows 6+ fractional bits gives >98% accuracy)

FRAC_BITS   = 6
SCALE       = 2 ** FRAC_BITS   # = 64

# Load the best saved model
model.load_state_dict(torch.load(CONFIG["save_path"]))
model.eval()

print("\nExporting weights as fixed-point integers...")
weight_dict = {}

for name, param in model.named_parameters():
    # Convert float weights → fixed-point integers
    fp_weights = (param.detach().cpu().numpy() * SCALE).astype(int)
    weight_dict[name] = fp_weights
    print(f"  {name:30s} shape={str(param.shape):20s} "
          f"min={fp_weights.min():5d}  max={fp_weights.max():5d}")

# Save as numpy file (can be read in FPGA tools)
np.save("lenet5_fixed_point_weights.npy", weight_dict)
print("\nFixed-point weights saved to lenet5_fixed_point_weights.npy")
print("These weights are ready to be loaded into FPGA ROM/RAM.")


# ─────────────────────────────────────────────
# STEP 9 — Per-Class Accuracy Report
# ─────────────────────────────────────────────

def per_class_accuracy(model, loader, device):
    model.eval()
    class_correct = [0] * 10
    class_total   = [0] * 10

    with torch.no_grad():
        for images, labels in loader:
            images  = images.to(device)
            labels  = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label]   += 1

    print("\nPer-Class Accuracy on Test Set:")
    print("-" * 35)
    for i in range(10):
        acc = 100.0 * class_correct[i] / class_total[i]
        print(f"  Digit {i}: {acc:.2f}%  ({class_correct[i]}/{class_total[i]})")

per_class_accuracy(model, test_loader, device)