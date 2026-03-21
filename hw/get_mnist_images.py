"""
get_mnist_images.py
────────────────────
Downloads MNIST test set and saves one clean example of each digit
we need: 1, 2, 3, 6.

Run from hw/ folder:
    python get_mnist_images.py

Requires: pip install torch torchvision pillow
"""
import torchvision
import os

# Which digits we need and what to name the files
NEEDED = {1: "1.png", 2: "2.png", 3: "3.png", 6: "6.png"}

print("Downloading MNIST test set (or using cached copy)...")
ds = torchvision.datasets.MNIST(
    root="./mnist_data",
    train=False,
    download=True
)

found = {}
for img, label in ds:
    if label in NEEDED and label not in found:
        fname = NEEDED[label]
        img.save(fname)
        found[label] = fname
        print(f"  Saved digit {label} → {fname}")
    if len(found) == len(NEEDED):
        break

print()
print("Done. Files saved:")
for digit, fname in NEEDED.items():
    exists = os.path.exists(fname)
    print(f"  {fname}  ({'OK' if exists else 'MISSING'})")

print()
print("Now run:")
print("  python prepare_4digits.py")
