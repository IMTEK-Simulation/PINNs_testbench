from tensorflow import convert_to_tensor, float32
import tensorflow as tf
import numpy as np


# Calculate the analytical solution for the set points
def simp_sol(inp):
    numer = np.exp(-inp) * (np.exp(2) - np.exp(2*inp))
    denom = np.exp(2) - 1
    return numer/denom


# Generate data points in a range as tf tensores
def gen_data(start, end, num):
    # generate the data pouits and reshape them into a row vector
    nums = np.linspace(start, end, num).reshape(-1, 1)
    return convert_to_tensor(nums, dtype=float32)


# Convert a set of boundary values to tf tensors
def set_boundaries(x_bc, y_bc):
    x = convert_to_tensor(x_bc, dtype=float32)
    y = convert_to_tensor(y_bc, dtype=float32)
    return x, y


# Simulate the ODE accross the given discretized range
def simp_sim(disc_x, y_bc, noise_level=0.02):
    # improves speed
    if isinstance(disc_x, tf.Tensor):
        disc_x = disc_x.numpy()

    # calculate the discretization step
    del_x = disc_x[1] - disc_x[0]
    num_points = len(disc_x)

    # initialze the solution vector
    u = np.zeros(num_points)

    for _ in range(80 * num_points):
        # enforce the boundary conditions in the solution
        u[0] = y_bc[0][0]
        u[-1] = y_bc[1][0]

        u_cp = u.copy()

        for i in range(1, num_points - 1):
            u[i] = (u_cp[i + 1] + u_cp[i - 1]) / (del_x**2 + 2)

    # add some additional noise to make it more escentric
    noise = noise_level * np.random.normal(0, 0.5, size=u.shape)
    u_noisy = u + noise

    return u, u_noisy
