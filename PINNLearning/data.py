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
    # generate the data points and reshape them into a row vector
    nums = np.linspace(start, end, num).reshape(-1, 1)
    return convert_to_tensor(nums, dtype=float32)


# Generate random 2D data points in a range as tf tensores
def gen_rand_2D_data(start, end, num):
    # Smallest possible float increment
    eps = np.finfo(float).eps

    # generate the data points and ensure that the lower
    # end cannot be part of the random interval
    nums1 = np.random.uniform(start + eps, end, num)
    nums2 = np.random.uniform(start + eps, end, num)

    # Combine into a single array
    n1n2 = np.column_stack((nums1, nums2))
    return convert_to_tensor(n1n2, dtype=float32)


# Convert a numpy array into a fitting tensor
def conv_data(np_array):
    # Test if the data is oriented in column first fashion
    if np_array.shape[0] < np_array.shape[-1]:
        print("WARNING: Watch out for the dimensions of the data vector.")
        print(f"-> The dimensions are: {np_array.shape}")

    return convert_to_tensor(np_array, dtype=float32)


# Returns a 2D tensor of the boundary conditions combined with another
# tensor marking the the second dimensions values.
def set_2D_boundaries(x_bc, y_bc, vec_sec_dim):
    # extend the boundary positon tensors to match the length of the vec_sec
    x_extended = [np.resize(i, (vec_sec_dim.shape[0], 1)) for i in x_bc]
    # join the arrays of the boundary positions with the vec_sec
    x_list = [tf.concat([conv_data(i), vec_sec_dim], axis=1)
              for i in x_extended[:2]]
    # if boundary conditions for multiple dimensions are given add them as well
    if len(x_bc) > 2:
        # remove the first and last elememt to prevent double assiging corners
        z_list = [tf.concat([vec_sec_dim, conv_data(i)], axis=1)[1:-1]
                  for i in x_extended[2:]]
        x_list += z_list
    # join the upper and lower bounds to reform the position vector
    x = tf.concat(x_list, axis=0)

    # extend the boundary value tensors to match the length of the vec_sec
    # if the nummber of conditions are >2 then reduce the dimension to match x
    y_extended = [tf.constant(i, shape=(vec_sec_dim.shape[0] - int(idx>=2) * 2, 1))
                  for idx, i in enumerate(y_bc)]
    # join the values for the upper and lower positons
    y = tf.concat(y_extended, axis=0)

    return x, y


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
def simp_sim(disc_x, y_bc, threshold=1e-6, noise_level=0.02):
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

    for _ in range(80 * num_points):  # scale for convergence
        u_cp = u.copy()

        for i in range(1, num_points - 1):
            u[i] = (u_cp[i + 1] + u_cp[i - 1]) / (del_x**2 + 2)

        # Check convergence, stop early if possible
        diff = np.max(np.abs(u - u_cp))
        if diff < threshold:
            break

    # add some additional noise to make it more escentric
    u_noisy = add_noise(u, noise_level)

    # reshape into the same data format as the gen_data
    # --> as these values will not be directly entered into a model,
    # they dont need to be a TF tensor
    return u.reshape(-1, 1), u_noisy.reshape(-1, 1)


# Simulate the ODE accross the given inhomogeneous discretized range
def alph_sim(disc_x, y_bc, alph, threshold=1e-6):
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

    for _ in range(80 * num_points):  # scale for convergence
        u_cp = u.copy()

        for i in range(1, num_points - 1):
            alph_r = alph((i + 0.5) * del_x)
            alph_l = alph((i - 0.5) * del_x)

            u[i] = (alph_r * u_cp[i + 1] + alph_l * u_cp[i - 1]) \
                / (del_x**2 + alph_l + alph_r)

        # Check convergence, stop early if possible
        diff = np.max(np.abs(u - u_cp))
        if diff < threshold:
            break

    # reshape into the same data format as the gen_data
    # --> as these values will not be directly entered into a model,
    # they dont need to be a TF tensor
    return u.reshape(-1, 1)


# Simulate the time influenced PDE accross the material
def time_sim(disc_x, y_bc, init_sol, max_sim_time, threshold=1e-6):
    # improves speed as numpy doesnt like Tf tensors
    if isinstance(disc_x, tf.Tensor):
        disc_x = disc_x.numpy().reshape(-1,)

    # calculate the discretization step in spatial domain
    del_x = disc_x[1] - disc_x[0]

    # calculate the discretization step in temporal domain
    del_t = del_x**2
    # FYI: set to ensure r = alpha*del_t/del_x^2 <= 1/2 (with alpha=0.1)
    time_steps = int(max_sim_time // del_t)
    r = 0.1 * del_t / del_x**2

    # initialze the solution vector with the initial solution
    u = init_sol(disc_x)
    # set up a history for the time steps of the solution
    u_hist = [u.copy()]

    # enforce the boundary conditions in the solution
    # as the iteration below doesnt cover these points,
    # they dont need to be reset
    u[0] = y_bc[0][0]
    u[-1] = y_bc[1][0]

    for t in range(time_steps):
        u_cp = u.copy()
        u[1:-1] = r * (u_cp[2:] + u_cp[:-2] - 2 * u_cp[1:-1]) + u_cp[1:-1]
        # FYI: This is an alternative and more performant approach to the
        # for loop from the other simulation function
        u_hist.append(u.copy())

        # Check convergence, stop early if possible
        diff = np.max(np.abs(u - u_cp))
        if diff < threshold:
            # print simulation information
            print(f"Converged at time t={t*del_t:.5f}s in {t} steps.")
            break

    # reshape into the same data format as the gen_data
    # --> as these values will not be directly entered into a model,
    # they dont need to be a TF tensor
    return u.reshape(-1, 1), u_hist, (t*del_t, max_sim_time)


# a
def twoD_sim(disc_x, y_bc, threshold=1e-6):
    # improves speed as numpy doesnt like Tf tensors
    if isinstance(disc_x, tf.Tensor):
        disc_x = disc_x.numpy().reshape(-1,)

    # set alpha = 0.1
    a = 0.1

    # calculate the discretization step
    del_xz = disc_x[1] - disc_x[0]
    num_points = len(disc_x)

    # precomute the denominator
    denom = 1 + 2 * a * (1/del_xz**2 + 1/del_xz**2)

    # initialze the solution vector
    u = np.zeros((num_points, num_points))

    # enforce the Dirichlet boundary conditions
    u[0, :] = y_bc[0][0]
    u[-1, :] = y_bc[1][0]

    # set initial Neumann boundary conditions
    u[1:-1, 0] = -y_bc[2][0] * del_xz
    u[1:-1, -1] = y_bc[3][0] * del_xz

    for _ in range(80 * num_points):  # scale for convergence
        u_cp = u.copy()
        u[1:-1, 1:-1] = (a * (
            (u[2:, 1:-1] + u[:-2, 1:-1]) / del_xz**2 +
            (u[1:-1, 2:] + u[1:-1, :-2]) / del_xz**2
            )
        ) / denom

        # enforce Neumann boundary conditions
        u[1:-1, 0] = u[1:-1, 1] - y_bc[2][0] * del_xz
        u[1:-1, -1] = u[1:-1, -2] + y_bc[3][0] * del_xz

        # Check convergence, stop early if possible
        diff = np.max(np.abs(u - u_cp))
        if diff < threshold:
            break

    # transpose the result to recover the correct orientaton
    # of the solution
    return u.T
