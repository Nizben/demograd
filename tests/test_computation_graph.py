import numpy as np
from demograd.tensor_engine import Tensor
from demograd.functions import *
from demograd.activations import *


def test_diamond_graph():
    # Test a diamond-shaped graph: d = (a + b) * a
    a = Tensor([2.0], requires_grad=True, name="a")
    b = Tensor([3.0], requires_grad=True, name="b")
    c = a + b  # c = a + b
    d = c * a  # d = c * a
    d.backward()

    # Expected gradients:
    # dd/da = c + a = (2+3) + 2 = 7
    # dd/db = a = 2
    assert np.allclose(a.grad, [7.0]), f"Expected a.grad=7.0, got {a.grad}"
    assert np.allclose(b.grad, [2.0]), f"Expected b.grad=2.0, got {b.grad}"


def test_reused_tensor():
    # Test reusing a tensor in multiple operations
    a = Tensor([2.0], requires_grad=True, name="a")
    b = a * a  # b = a^2
    c = b * a  # c = a^3
    c.backward()
    assert np.allclose(a.grad, [3 * 2**2]), f"Expected 12.0, got {a.grad}"


if __name__ == "__main__":
    test_diamond_graph()
    test_reused_tensor()
    print("All computation graph tests passed!")
