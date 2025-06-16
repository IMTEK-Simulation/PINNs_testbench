from tensorflow import GradientTape, reduce_mean, square
from keras import optimizers

# FYI: All implementations here are made to be flexible and support various
# different use-cases. This is typically not necessary, as such the code
# could can be largely simplified.


# Implementation of the initially simplified ODE as the loss
# based on the residual and the boundary conditions
def oneD_loss(model, inp, x_bc, y_bc, alph=None):
    # Get the "output" from the model as values of the target function
    # and calculate the derivates with respect to the input features
    with GradientTape(persistent=True) as tape:
        tape.watch(inp)
        y_pred = model(inp)

        # the tape records all operations -->
        # only call gradient inside of the tape if you need a higher order
        y_x = tape.gradient(y_pred, inp)
    y_xx = tape.gradient(y_x, inp)
    del tape

    if alph is not None:
        # calculate the residual of the ode
        # for inhomogenuous material distribution
        residual = y_pred - alph(inp) * y_xx
    else:
        # calculate the residual of the ode
        # for a homogenuous material
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


# Define an learning rate schedule to improve convergence
def learning_rate_schedule(init, steps, rate):
    return optimizers.schedules.ExponentialDecay(
        initial_learning_rate=init,
        decay_steps=steps,
        decay_rate=rate
    )


# Implementing the training function
def train(model, x_train, x_bc, y_bc, loss_func, lr_schedule=None,
          limit=3500, threshold=1e-9, write=True):
    loss_time = []

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
        # calculate the forward pass
        with GradientTape() as tape:
            loss = loss_func(model, x_train, x_bc, y_bc)

        # calculate the backwards pass
        # first calculate the gradients
        grads = tape.gradient(loss, model.trainable_variables)
        # let optimizer adjust values
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # update all the variables
        delta_loss = last_loss - loss
        last_loss = loss
        epoch += 1
        loss_time.append((epoch, loss))

        # stop if it takes too long
        if epoch >= limit:
            print("DID NOT CONVERGE!")
            break
        # print current state
        elif epoch % 100 == 0 and write:
            print(f"Epoch {epoch}: Loss = {loss}")
    print(f"Last Epoch {epoch}: last Loss = {loss}")

    return loss_time
