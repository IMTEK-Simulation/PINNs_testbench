from tensorflow import GradientTape, reduce_mean, square
from keras import optimizers
from data import simp_ode
import tensorflow as tf


# Compute the loss over the ode and the boundary conditions
def simp_loss(model, inp, x_bc, y_bc):
    residual = simp_ode(model, inp)
    # compute mean squares error of the residual
    loss_pde = reduce_mean(square(residual))

    # predict the values for the boundaries
    y_bc_pred = model(x_bc)
    # calculate the mean squared error of the boundaries
    loss_bc = reduce_mean(square(y_bc - y_bc_pred))

    return loss_pde + loss_bc


# Define an learning rate schedule to improve convergence
def learning_rate_schedule(init, steps, rate):
    return optimizers.schedules.ExponentialDecay(
        initial_learning_rate=init,
        decay_steps=steps,
        decay_rate=rate
    )


# Implementing the training step, use decorator to increase efficiancy
# @tf.function
def train(model, x_train, x_bc, y_bc, lr_schedule=None, threshold=1e-9, write=True):
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
            loss = simp_loss(model, x_train, x_bc, y_bc)

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
        if epoch % 100 == 0 and write:
            print(f"Epoch {epoch}: Loss = {loss}")
    print(f"Last Epoch {epoch}: last Loss = {loss}")

    return loss_time
