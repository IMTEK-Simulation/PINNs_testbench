# PINN Testbench

A testing ground for different implementations of physics informed neural networks (PINNs) for the purpose of solving differential equations with variing complexity.
The various implementations are included as jupyter notebooks in the `examples` subdirectory. In `PINNLearning` the corresponding functions are implemented.
The implementations are based on the [Tensorflow](https://www.tensorflow.org) and [Keras](https://keras.io) packages. However they can also be converted with minor effort to [PyTorch](https://pytorch.org).


## Tests

Before being able to run tests, you need to execute
```python
pip install -e .[test] 
```
to editably install the code.