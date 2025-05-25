from tensorflow import convert_to_tensor, float32
import numpy as np


# Calculate the analytical solution for the set points
def simp_sol(inp):
    numer = np.exp(-inp) * (np.exp(2) - np.exp(2*inp))
    denom = np.exp(2) - 1
    return numer/denom


# Generate data points in a range as tf tensores
def gen_data(start, end, num):
    nums = np.linspace(start, end, num).reshape(-1, 1)
    return convert_to_tensor(nums, dtype=float32)


# Convert a set of boundary values to tf tensors
def set_boundaries(x_bc, y_bc):
    x = np.array(x_bc).reshape(-1, 1)
    x = convert_to_tensor(x, dtype=float32)

    y = np.array(y_bc).reshape(-1, 1)
    y = convert_to_tensor(y, dtype=float32)

    return x, y
