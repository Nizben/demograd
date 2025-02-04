import numpy as np
from demograd.tensor_engine import Tensor
from demograd.functions import *
from demograd.activations import *


def test_no_grad():
    a = Tensor([2.0], requires_grad=False)
    b = Tensor([3.0], requires_grad=False)
    c = a + b
    c.backward()  # Should do nothing
    assert a.grad is None, "Grad should not exist for requires_grad=False"


if __name__ == "__main__":
    test_no_grad()
    print("All edge case tests passed!")
