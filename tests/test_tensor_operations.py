import numpy as np
from demograd.tensor_engine import Tensor
from demograd.functions import *
from demograd.activations import *


def test_addition():
    a = Tensor([2.0], requires_grad=True, name="a")
    b = Tensor([3.0], requires_grad=True, name="b")
    c = a + b
    c.backward()
    assert np.allclose(a.grad, [1.0]), f"Expected a.grad=1.0, got {a.grad}"
    assert np.allclose(b.grad, [1.0]), f"Expected b.grad=1.0, got {b.grad}"


def test_multiplication():
    a = Tensor([2.0], requires_grad=True, name="a")
    b = Tensor([3.0], requires_grad=True, name="b")
    c = a * b
    c.backward()
    assert np.allclose(a.grad, [3.0]), f"Expected a.grad=3.0, got {a.grad}"
    assert np.allclose(b.grad, [2.0]), f"Expected b.grad=2.0, got {b.grad}"


def test_subtraction():
    a = Tensor([5.0], requires_grad=True, name="a")
    b = Tensor([3.0], requires_grad=True, name="b")
    c = a - b
    c.backward()
    assert np.allclose(a.grad, [1.0]), f"Expected a.grad=1.0, got {a.grad}"
    assert np.allclose(b.grad, [-1.0]), f"Expected b.grad=-1.0, got {b.grad}"


if __name__ == "__main__":
    test_addition()
    test_multiplication()
    test_subtraction()
    print("All tensor operation tests passed!")
