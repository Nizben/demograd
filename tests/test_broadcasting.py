import numpy as np
from demograd.tensor_engine import Tensor
from demograd.functions import *
from demograd.activations import *


def test_broadcast_add():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # Shape (2, 2)
    b = Tensor([10.0, 20.0], requires_grad=True)  # Shape (2,)
    c = a + b  # Broadcasts b to (2, 2)
    c.backward(grad=np.ones_like(c.data))
    assert np.allclose(a.grad, [[1.0, 1.0], [1.0, 1.0]]), "Broadcast add grad a failed"
    assert np.allclose(b.grad, [2.0, 2.0]), "Broadcast add grad b failed"


def test_broadcast_mul():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # Shape (2, 2)
    b = Tensor([10.0], requires_grad=True)  # Shape (1,)
    c = a * b  # Broadcasts b to (2, 2)
    c.backward(grad=np.ones_like(c.data))
    assert np.allclose(
        a.grad, [[10.0, 10.0], [10.0, 10.0]]
    ), "Broadcast mul grad a failed"
    assert np.allclose(b.grad, [10.0]), "Broadcast mul grad b failed"


def test_broadcast_div():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # Shape (2, 2)
    b = Tensor([10.0], requires_grad=True)  # Shape (1,)
    c = a / b  # Broadcasts b to (2, 2)
    c.backward(grad=np.ones_like(c.data))
    assert np.allclose(a.grad, [[0.1, 0.1], [0.1, 0.1]]), "Broadcast div grad a failed"
    assert np.allclose(b.grad, [-0.1]), "Broadcast div grad b failed"


def test_broadcast_sub():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # Shape (2, 2)
    b = Tensor([10.0], requires_grad=True)  # Shape (1,)
    c = a - b  # Broadcasts b to (2, 2)
    c.backward(grad=np.ones_like(c.data))
    assert np.allclose(a.grad, [[1.0, 1.0], [1.0, 1.0]]), "Broadcast sub grad a failed"
    assert np.allclose(b.grad, [-4.0]), "Broadcast sub grad b failed"


def test_broadcast_pow():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # Shape (2, 2)
    b = Tensor([2.0], requires_grad=True)  # Shape (1,)
    c = a**b  # Broadcasts b to (2, 2)
    c.backward(grad=np.ones_like(c.data))
    assert np.allclose(a.grad, [[2.0, 4.0], [6.0, 8.0]]), "Broadcast pow grad a failed"
    assert np.allclose(b.grad, [34.84], atol=1e-3), "Broadcast pow grad b failed"


def test_broadcast_relu():
    a = Tensor([[1.0, -2.0], [-3.0, 4.0]], requires_grad=True)  # Shape (2, 2)
    c = ReLU.apply(a)
    c.backward(grad=np.ones_like(c.data))
    assert np.allclose(a.grad, [[1.0, 0.0], [0.0, 1.0]]), "Broadcast relu grad failed"


if __name__ == "__main__":
    test_broadcast_add()
    test_broadcast_mul()
    test_broadcast_div()
    test_broadcast_sub()
    test_broadcast_pow()
    test_broadcast_relu()
    print("All tests passed!")
