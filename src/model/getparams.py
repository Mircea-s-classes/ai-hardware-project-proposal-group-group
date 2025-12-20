import numpy as np

data = np.load("norm_params.npz")

print("x_mean =", data["x_mean"])
print("x_std  =", data["x_std"])
print("y_mean =", data["y_mean"])
print("y_std  =", data["y_std"])
