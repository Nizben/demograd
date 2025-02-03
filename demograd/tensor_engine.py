import numpy as np
from . import utils
from demograd import functions

# Basic tensor class
class Tensor:
    def __init__(self, data, requires_grad=False, depends_on=None, name=None):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad  # Whether to compute gradients : boolean

        # List of dependencies (parents in the computational graph),  if depends_on is None, then it is an empty list
        self.depends_on = depends_on or []
        # Function that is creating the tensor
        self._grad_fn = None
        # For naming tensors (useful for debugging)
        self.name = name

    def set_grad_fn(self, grad_fn):
        self._grad_fn = grad_fn

    def backward(self, grad=None):
        if not self.requires_grad:
            return

        if grad is None:
            if self.data.size == 1:
                grad = np.ones_like(self.data)
            else:
                raise RuntimeError("grad must be specified for non-scalar tensors.")

        self.grad = grad

        # Perform topological sort to order the functions
        topo_order = utils.topological_sort(self)

        for function in reversed(topo_order):
            grads = function.backward(function.output.grad)
            for inp, g in zip(function.inputs, grads):
                if inp.requires_grad:
                    if inp.grad is None:
                        inp.grad = g
                    else:
                        inp.grad = inp.grad + g

    # Overloading operations
    def __add__(self, other):
        return functions.Add.apply(self, other)

    def __radd__(self, other):
        return functions.Add.apply(self, other)

    def __sub__(self, other):
        return functions.Sub.apply(self, other)

    def __rsub__(self, other):
        return functions.Sub.apply(other, self)

    def __mul__(self, other):
        return functions.Mul.apply(self, other)

    def __rmul__(self, other):
        return functions.Mul.apply(self, other)
    
    def __truediv__(self, other):
        return functions.Div.apply(self, other)
    
    def __rtruediv__(self, other):
        return functions.Div.apply(other, self)
    
    def __neg__(self):
        return functions.Neg.apply(self)
    
    def __pow__(self, other):
        return functions.Pow.apply(self, other)
    
    def __rpow__(self, other):
        return functions.Pow.apply(other, self)

    def __exp__(self):
        return functions.Exp.apply(self)

    def __getitem__(self, key):
        # Slicing operation
        sliced_data = self.data[key]
        new_tensor = Tensor(sliced_data, requires_grad=self.requires_grad, depends_on=self.depends_on)
        return new_tensor

    # Representation
    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"