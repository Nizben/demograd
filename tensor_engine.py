import numpy as np 
from functions import Function
from utils import topological_sort

# Basic tensor class
class Tensor:
    def __init__(self, data, requires_grad = False, depends_on = None, name = None):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad

        # List of dependencies (parents in the computational graph)
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
        topo_order = topological_sort(self)
        
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
        from functions import Add
        return Add.apply(self, other)

    def __radd__(self, other):
        from functions import Add
        return Add.apply(self, other)

    def __sub__(self, other):
        from functions import Sub
        return Sub.apply(self, other)

    def __rsub__(self, other):
        from functions import Sub
        return Sub.apply(other, self)

    def __mul__(self, other):
        from functions import Mul
        return Mul.apply(self, other)

    def __rmul__(self, other):
        return Mul.apply(self, other)

    
    

    # Representation
    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    


print('done')