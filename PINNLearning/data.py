from tensorflow import convert_to_tensor, float32
import tensorflow as tf
import numpy as np


# Calculate the analytical solution for the set points
def simp_sol(inp):
    numer = np.exp(-inp) * (np.exp(2) - np.exp(2*inp))
    denom = np.exp(2) - 1
    return numer/denom


# Calculate the analytical solution in inhomogenous material
def simp_sol_var_mat(inp, alph):
    scal = np.sqrt(1/alph(inp))
    numer = np.exp(-inp*scal) * (np.exp(2*scal) - np.exp(2*scal*inp))
    denom = np.exp(2*scal) - 1
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


# Add random, normally distributed noise to a numpy array
def add_noise(arr, noise_level, end_values=True, bounds=None):
    noise = noise_level * np.random.normal(0, 0.5, size=arr.shape)
    if not end_values:
        noise[0] = 0
        noise[-1] = 0
    arr_noisy = arr + noise

    if bounds is not None:
        arr_noisy = np.clip(arr_noisy, bounds[0], bounds[1])
    return arr_noisy


# Simulate the ODE accross the given discretized range
def simp_sim(disc_x, y_bc, noise_level=0.02):
    # improves speed as numpy doesnt like Tf tensors
    if isinstance(disc_x, tf.Tensor):
        disc_x = disc_x.numpy().reshape(-1,)

    # calculate the discretization step
    del_x = disc_x[1] - disc_x[0]
    num_points = len(disc_x)

    # initialze the solution vector
    u = np.zeros(num_points)

    # enforce the boundary conditions in the solution
    # as the iteration below doesnt cover these points,
    # they dont need to be reset
    u[0] = y_bc[0][0]
    u[-1] = y_bc[1][0]

    for _ in range(80 * num_points):  # important for convergence
        u_cp = u.copy()

        for i in range(1, num_points - 1):
            u[i] = (u_cp[i + 1] + u_cp[i - 1]) / (del_x**2 + 2)

    # add some additional noise to make it more escentric
    u_noisy = add_noise(u, noise_level)

    # reshape into the same data format as the gen_data
    # --> as these values will not be directly entered into a model,
    # they dont need to be a TF tensor
    return u.reshape(-1, 1), u_noisy.reshape(-1, 1)
