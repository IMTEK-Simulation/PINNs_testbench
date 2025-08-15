# PINN Testbench

This repository contains a series of Jupyter Notebooks that explore the application of machine learning methods for solving differential equations. We provide well-documented code for the manual implementation of machine learning models, alongside illustrations of using frameworks. Each problem is accompanied by some theoretical explanations regarding the behavior of the algorithms, as well as the reasoning behind the implementations and results.


The primary focus of this work is on **Physics-Informed Neural Networks (PINNs)**. To introduce this technique, we utilize the heat equation with material-dependent diffusivity:

$$\dfrac{\partial u(\vec{x}, t)}{\partial t} = \dfrac{\partial}{\partial \vec{x}} \Bigg[ \alpha(\vec{x}) \cdot \dfrac{\partial u(\vec{x}, t)}{\partial \vec{x}}\Bigg]$$


We begin by solving a significantly simplified version of this equation, gradually incorporating complexities to introduce key concepts. The series of notebooks covers various topics, including a quantitative analysis of learning from noisy measurement data, compensation for incomplete problem formulations, modifications for learning multidimensional and time-dependent equations, solving the inverse problem for an unknown $\alpha(\vec{x})$, and learning solutions under varying boundary conditions.


Additionally, we discuss the theoretical foundations of **Neural Operators** to provide a overview of this general field. A brief introduction to the critical topics of **hyperparameter optimization (HPO)** and **neural architecture search (NAS)** is also included. While these topics are not strictly necessary for the problems addressed here, they offer a broader perspective on strategies for tackling larger and more complex challenges.


## Notebooks

The problems are structured, beginning with the significantly simplified initial setup presented in the notebook titled `1D & time independent`. The complexity of the equations is gradually increased until they align with the general problem description outlined above, at least in _one-dimensional space_ and without considering _time dependence_. The latter two modifications are explicitly presented in the `2D & time dependent` notebook, culminating in a solution to the full problem as described.

For manual implementations, the `PINNLearning` library was developed to streamline the notebook structures. The exploration of **HPO** and **NAS** is provided in a dedicated notebook. The final extensions cover special adaptations to the _"classic"_ PINN problems and include the discussion of **neural operators** in the `Special Problems` notebook.

## Implementation

The implementations primarily utilize the [TensorFlow](https://www.tensorflow.org) framework, which is the industry standard for production. Manual implementations are conducted using the integrated [Keras](https://keras.io) package. However, all implementations can be easily adapted to the scientific standard [PyTorch](https://pytorch.org) framework with minimal effort.

Each problem is configured to pull a pre-trained model from the data folder. This behavior can be manually disabled for each problem, allowing the respective model to be retrained and overwrite the saved version. A cautionary note: some learning algorithms (as indicated in the theoretical section of each problem) are resource-intensive and may take a considerable amount of time to compute, potentially freezing your computer. It is advisable to utilize a server cluster for running these algorithms.

Before running the notebooks, ensure that all necessary packages listed in the `pyproject.toml` file and their dependencies are installed. You can use tools like Poetry, pipenv or uv to directly install from the file.

## Tests

Before being able to run tests, you need to execute
```python
pip install -e .[test] 
```
to editably install the code.
