from keras import Sequential, layers, regularizers

# FYI: The implementation here is made to be flexible and support various
# different use-cases. This is typically not necessary, as such the code
# could can be largely simplified.


# Function to dynamically create the model with adjustable number of hidden
# layers and weights per hidden layer
def create_model(num_layers, weights, out_dim=1, l2=0.01):
    # Check for a minimum number of hiddenlayers
    if num_layers < 1:
        return None

    # Ensure that the number of weights is a list corresponding to each layer
    if isinstance(weights, int):
        weights = [weights] * num_layers
    elif len(weights) < num_layers:
        dif = num_layers - len(weights)
        li = [weights[-1]] * dif
        weights.append(li)

    # Dynamically creating the model, enforce the input dimension of 1
    model = []
    for i in range(num_layers):
        model.append(layers.Dense(weights[i], activation='tanh',
                                  kernel_regularizer=regularizers.l2(l2)))
    model.append(layers.Dense(out_dim))
    return Sequential(model)
