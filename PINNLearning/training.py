from tensorflow import GradientTape, reduce_mean, square, constant, \
    tile, shape, data
from keras import optimizers

# FYI: All implementations here are made to be flexible and support various
# different use-cases. This is typically not necessary, as such the code
# could can be largely simplified.


# Implementation of the initially simplified ODE as the loss
# based on the residual and the boundary conditions
def oneD_loss(model, inp, x_bc, y_bc, alph=None):
    # get the "output" from the model as values of the target function
    # and calculate the derivates with respect to the input features
    with GradientTape() as tape2:
        tape2.watch(inp)
        with GradientTape() as tape1:
            tape1.watch(inp)
            y_pred = model(inp)
        # FYI: The tape records all operations, allowing automatic
        # differentiation. You may call gradient inside of a tape, but
        # only if you need a higher order without having the capacity
        # to do the nested tapes. This is not best-practice however.

        y_x = tape1.gradient(y_pred, inp)
        if alph is not None:
            # adjust the derivative for a inhomogenuous
            # material distribution
            y_x = alph(inp) * y_x
    y_xx = tape2.gradient(y_x, inp)

    # calculate the residual of the ode
    residual = y_pred - y_xx
    # compute mean squares error of the residual
    loss_pde = reduce_mean(square(residual))

    # predict the values for the boundaries
    y_bc_pred = model(x_bc)
    # calculate the mean squared error of the boundaries
    loss_bc = reduce_mean(square(y_bc - y_bc_pred))

    # --- IMPORTATNT ---
    # Watch out, that the loss terms are always of the same order!
    # Here given automatically, thrugh choosing the problem setting
    # to be bewteen 0 and 1, with the bc as 0 and 1
    return loss_pde + loss_bc


# Implementation of the simplified but time dependent PDE as the loss
# based on the residual and the boundary/ initial conditions
def time_loss(model, inp, x_bc, y_bc, init_sol):
    # here the inp is split in an tensor in the form of [x, t]
    with GradientTape() as tape2:
        tape2.watch(inp)
        with GradientTape() as tape1:
            tape1.watch(inp)
            y_pred = model(inp)

        # calculate the gradients with respect to both input variables
        grads = tape1.gradient(y_pred, inp)
        # split the calculated gradients of x and t
        y_x = grads[:, 0:1]
        y_t = grads[:, 1:2]
    # calculate the second derivative of x
    y_xx = tape2.gradient(y_x, inp)
    y_xx = y_xx[:, 0:1]

    # calculate the residual of the pde
    residual = y_t - 0.1 * y_xx
    # compute mean squares error of the residual
    loss_pde = reduce_mean(square(residual))

    # predict the values at the boundarys - here x_bc is 2D
    y_bc_pred = model(x_bc)
    # calculate the mean squared error of the boundaries
    loss_bc = reduce_mean(square(y_bc - y_bc_pred))

    # create a mask with ones in the first column and zeros in the second
    mask = constant([[1.0, 0.0]], dtype=inp.dtype)
    mask = tile(mask, [shape(inp)[0], 1])
    # predict the values for the initial solution at t=0
    y_init_pred = model(inp * mask)
    # calculate the inital solution at the x values of the input data and get
    # the mean squared error of the initial solution
    init_vals = init_sol(inp[:, 0:1])
    loss_init = reduce_mean(square(init_vals - y_init_pred))

    return loss_pde + loss_bc + loss_init


# Define an learning rate schedule to improve convergence
def learning_rate_schedule(init, steps, rate):
    return optimizers.schedules.ExponentialDecay(
        initial_learning_rate=init,
        decay_steps=steps,
        decay_rate=rate
    )


# Split the x_train set into batches of the size batch_size and enable the
# distribution of these batches across the CPU
def data_batch(data_set, batch_size):
    # create a tf dataset fromt the given tf tensor, shuffle and batch it
    data_batched = data.Dataset.from_tensor_slices(data_set)
    data_batched = data_batched.shuffle(buffer_size=1024)
    data_batched = data_batched.batch(batch_size)
    data_batched = data_batched.prefetch(data.AUTOTUNE)
    return data_batched


# Definition of the inidvidual steps in training a NN
def train_step(model, x_train, x_bc, y_bc, loss_func, optimizer):
    # calculate the forward pass
    with GradientTape() as tape:
        loss = loss_func(model, x_train, x_bc, y_bc)

    # calculate the backwards pass
    # first calculate the gradients
    grads = tape.gradient(loss, model.trainable_variables)
    # let optimizer adjust values
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


# Implementing the training function
def train(model, x_train, x_bc, y_bc, loss_func, lr_schedule=None,
          limit=3500, threshold=1e-9, batch=None, write=True):
    loss_time = []

    # enable batching of x_train
    if batch is not None:
        x_batched = data_batch(x_train, batch)

    # enable setting of a learning rate scheduler
    if lr_schedule is not None:
        optimizer = optimizers.Adam(learning_rate=lr_schedule)
    else:
        optimizer = optimizers.Adam()

    # train the model until change in loss is below a threshold
    delta_loss = 1
    last_loss = 1
    epoch = 1
    while abs(delta_loss) > threshold:
        # call the train_step depending if x_train is batched or not
        if batch is not None:
            loss = 0
            for batch in x_batched:
                loss += train_step(model, batch, x_bc, y_bc, loss_func, optimizer)
        else:
            loss = train_step(model, x_train, x_bc, y_bc, loss_func, optimizer)

        # update all algorithm the variables
        delta_loss = last_loss - loss
        last_loss = loss
        epoch += 1
        loss_time.append((epoch, loss))

        # stop if it takes too long
        if epoch >= limit:
            print("BROKEN OFF TRANING: Did not converge in limit!")
            break
        # print current state
        elif epoch % 100 == 0 and write:
            print(f"Epoch {epoch}: Loss = {loss}")
    print(f"Last Epoch {epoch}: last Loss = {loss}")

    return loss_time
