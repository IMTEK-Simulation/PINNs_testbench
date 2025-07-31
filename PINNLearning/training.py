from tensorflow import GradientTape, reduce_mean, square, constant, \
    tile, shape, data, split, distribute, keras
import tensorflow as tf

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
            # adjust the derivative for a inhomogenuous material distribution
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


# Implementation of the 2D time independent PDE as the loss
# based on the residual and the Dirichlet/Neumann boundary conditions
def twoD_loss(model, inp, x_bc, y_bc):
    # here the inp consists of a tensor in the form of [x, z]
    with GradientTape(persistent=True) as tape2:
        tape2.watch(inp)
        with GradientTape(persistent=True) as tape1:
            tape1.watch(inp)
            y_pred = model(inp)

        # calculate the gradients with respect to both input variables
        grads = tape1.gradient(y_pred, inp)
        y_x = grads[:, 0:1]
        y_z = grads[:, 1:2]
    # split and calculate the second order gradients for x and z
    y_xx = tape2.gradient(y_x, inp)[:, 0:1]
    y_zz = tape2.gradient(y_z, inp)[:, 1:2]

    # remove the tapes from memory
    del tape1
    del tape2

    # calculate the residual of the ode
    residual = y_pred - 0.1 * (y_xx + y_zz)
    # compute mean squares error of the residual
    loss_pde = reduce_mean(square(residual))

    # split boundaries into Dirichlet and Neumann
    split_point = shape(x_bc)[0]//2
    x_bc_dir, x_bc_neu = split(x_bc,[split_point, shape(x_bc)[0]-split_point],axis=0)
    y_bc_dir, y_bc_neu = split(y_bc,[split_point, shape(x_bc)[0]-split_point],axis=0)

    # predict the values for the Dirichlet boundaries
    y_bc_dir_pred = model(x_bc_dir)
    # calculate the mean squared error of the Dirichlet boundaries
    loss_bc_dir = reduce_mean(square(y_bc_dir - y_bc_dir_pred))

    # Calculate the gradients for the Neumann boundaries
    with GradientTape() as tapeNeu:
        tapeNeu.watch(x_bc_neu)
        y_bc = model(x_bc_neu)
    y_z_bc_neu = tapeNeu.gradient(y_bc, x_bc_neu)[:, 1:2]
    # calculate the mean squared error of the Neumann boundaries
    loss_bc_neu = reduce_mean(square(y_bc_neu - y_z_bc_neu))

    return loss_pde + 2*loss_bc_dir + 2*loss_bc_neu


# Implementation of the simplified but time dependent PDE as the loss
# based on the residual and the boundary/ initial conditions
def time_loss(model, inp, x_bc, y_bc, init_sol):
    # here the inp consists of a tensor in the form of [x, t]
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

    # FYI: Here the losses are manually weighted. This is for the purpose of an
    # improved enforcement of the boundary and initial conditions, as well as a
    # compensation for the comparatively limited number of boundary points.
    return loss_pde + 2*loss_bc + 2*loss_init


# Define an learning rate schedule to improve convergence
def learning_rate_schedule(init, steps, rate):
    return keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=init,
        decay_steps=steps,
        decay_rate=rate
    )


# Split the x_train set into batches of the size batch_size and enable the
# distribution of these batches across the CPU
def data_batch(data_set, batch_size, strategy):
    # create a tf dataset fromt the given tf tensor, shuffle and batch it
    data_batched = data.Dataset.from_tensor_slices(data_set)
    data_batched = data_batched.shuffle(buffer_size=1024)
    data_batched = data_batched.batch(batch_size)
    data_batched = data_batched.prefetch(data.AUTOTUNE)
    return strategy.experimental_distribute_dataset(data_batched)


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


# A wrapper to improve the performance of parallelism in batching
@tf.function
def distributed_train_step(strategy, model, x_batched, x_bc, y_bc, loss_func, optimizer):
    # distribute the samples in one batch across the CPUs/GPUs
    per_replica_loss = strategy.run(
                    train_step, args=(model, x_batched, x_bc, y_bc, loss_func, optimizer)
                )
    # FYI: The strategy will compute the gradient for each subset of the batch
    # and averages it across all subsets. The update to the model is
    # synchronized and thus does not differ to the computation on a single device.

    # sum up the computed losses of all distributed instances
    batch_loss = strategy.reduce(distribute.ReduceOp.SUM, per_replica_loss, axis=None)

    return batch_loss


# Implementing the training function
def train(model, x_train, x_bc, y_bc, loss_func, lr_schedule=None,
          limit=3500, threshold=1e-9, batch=None, write=True):
    loss_time = []

    # batch x_train in size of 'batch' is set
    if batch is not None:
        # activate the distribution strategy for the batches
        strategy = distribute.MirroredStrategy()
        print('Number of devices:', strategy.num_replicas_in_sync)
        # batch the data
        x_batched = data_batch(x_train, batch, strategy)

        # enable maintenance of the sheduler across distributed instances
        with strategy.scope():
            # enable setting of a learning rate scheduler
            if lr_schedule is not None:
                optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
            else:
                optimizer = keras.optimizers.Adam()
    else:
        # enable setting of a learning rate scheduler
        if lr_schedule is not None:
            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        else:
            optimizer = keras.optimizers.Adam()

    # train the model until change in loss is below a threshold
    delta_loss = 1
    last_loss = 1
    epoch = 1
    while abs(delta_loss) > threshold:
        # call the train_step depending if x_train is batched or not
        if batch is not None:
            loss = 0
            for batch in x_batched:
                loss += distributed_train_step(strategy, model, batch, x_bc, 
                                               y_bc, loss_func, optimizer)
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
