import os

path = os.path.join(os.path.dirname(__file__), "weights.npz")
if not os.path.isfile(path):
    raise FileNotFoundError(f"weights.npz not found at {path}")

size = os.path.getsize(path)
print(f"weights.npz → {size} bytes ({size/1024:.1f} KB, {size/1024**2:.2f} MB)")
