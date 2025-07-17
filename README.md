# PINN Testbench

A testing ground for different implementations of physics informed neural networks (PINNs) for the purpose of solving differential equations with variing complexity.
The various implementations are included as jupyter notebooks. In the `PINNLearning` folder the corresponding functions are implemented.
The implementations are based on the [TensorFlow](https://www.tensorflow.org) and [Keras](https://keras.io) packages. However they can also be converted with minor effort to [PyTorch](https://pytorch.org).

## Notebooks

The initial problem setup and basic information about deep learning and PINNs is given in the notebook of `1D & time independent` problems.
More complicated problems are posed in `2D & time dependent`.
A excursion about the more in-depth but interesting topics of **hyperparameter optimization (HPO)** and **neural architeture search (NAS)** is provided in the corresponding notebook.

Finaly a special application of PINNs is considered in `Specific Problems`.


## Tests

Before being able to run tests, you need to execute
```python
pip install -e .[test] 
```
to editably install the code.