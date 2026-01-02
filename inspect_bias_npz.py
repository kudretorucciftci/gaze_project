import numpy as np

PROFILE_PATH = "implicit_bias.npz"

data = np.load(PROFILE_PATH)

bx = data["bx"]
by = data["by"]

print("Bias map shape:", bx.shape)
print()

print("bx stats:")
print("  min:", bx.min())
print("  max:", bx.max())
print("  mean:", bx.mean())
print()

print("by stats:")
print("  min:", by.min())
print("  max:", by.max())
print("  mean:", by.mean())
print()

print("Center cell (approx):")
c = bx.shape[0] // 2
print("  bx:", bx[c, c])
print("  by:", by[c, c])
