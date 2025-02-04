# Demograd

**Demograd** is a minimal autograd engine and neural network library built for educational purposes. It is designed to mimic the core functionalities of PyTorch, including a tensor class with automatic differentiation, a set of basic differentiable operations, activation functions, neural network layers, and optimizers. The design emphasizes clarity, modularity, and reproducibility.

> **Note:** This repository is still a work in progress!

> **Note:** This automatic differentiation engine is also heavily inspired by Karpathy's [Micrograd](https://github.com/karpathy/micrograd).

## Overview

**Demograd** provides the following core components:

- **Tensor and Autograd Engine:**  
  The `Tensor` class (in `tensor_engine.py`) encapsulates NumPy arrays along with gradient information and a dependency graph. It supports automatic differentiation via a topological sorting mechanism for the computational graph.

- **Differentiable Operations:**  
  A collection of basic operations (e.g., addition, subtraction, multiplication, division, exponentiation, logarithm, matrix multiplication) are implemented as subclasses of a base `Function` (in `functions.py`). Each operation defines a static `apply` method for the forward pass and a corresponding `backward` method for computing gradients.

- **Activation Functions:**  
  Common activation functions such as ReLU, Sigmoid, Tanh, and Softmax are provided in `activations.py`. These functions follow the same autograd pattern, allowing them to be used seamlessly in network architectures.

- **Neural Network Layers:**  
  A basic neural network module system is available in `nn.py`. This includes a `Linear` layer for fully connected networks and a `Sequential` container that aggregates multiple layers and collects their parameters.

- **Optimizers:**  
  Simple optimizers (SGD and Adam) are implemented in `optimizers.py`. They operate on the parameters of the network and provide `step()` and `zero_grad()` methods to update weights based on computed gradients.

- **Example Training Script:**  
  An example jupyter notebook (`example.ipynb`) demonstrates how to build and train a basic multilayer perceptron (MLP) on synthetic data using the provided modules.

- **Visualization:** 
  A computational graph building tool is also available in `visualization.py`. It provides the ability to visualize the computational graph of your modules and eases debugging and error tracking when building large networks.


## Usage

### Building Models

You can construct neural networks by composing layers defined in `nn.py`. For example, a simple MLP can be created as follows:

```python
from demograd.nn import Linear, Sequential
from demograd.activations import ReLU

# Define an MLP with one hidden layer:
model = Sequential(
    Linear(input_dim, hidden_dim),
    ReLU.apply,
    Linear(hidden_dim, output_dim)
)
```

## License
This project is provided for educational purposes. Use it freely to learn about autograd engines and neural network training.