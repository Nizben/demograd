import numpy as np 


# Basic tensor class
class Tensor:
    def __init__(self, data, requires_grad = False):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad

        # Function that is creating the tensor
        self._grad_fn = None

    def set_grad_fn(self, grad_fn):
        self._grad_fn = grad_fn     

    def backward(self, grad = None):
        if not self.requires_grad:
            return
        
        if grad is None:
            grad = np.ones_like(self.data)

        self.grad = grad

        functions = [self._grad_fn]

        while functions:
            function = functions.pop()
            if function is not None:
                grads = function.backward(function.output.grad)
                for inp, grad in zip (functions.inputs, grads):
                    if inp.requires_grad:
                        if inp.grad is None:
                            inp.grad = grad
                        else:
                            inp.grad += grad
                        if inp._grad_fn is not None:
                            functions.append(inp._grad_fn)

    # Overloading operations
    def __add__(self, other):
        return Add.apply(self, other)

    def __radd__(self, other):
        return Add.apply(self, other)

    def __mul__(self, other):
        return Mul.apply(self, other)

    def __rmul__(self, other):
        return Mul.apply(self, other)
    

    # Representation
    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    


print('done')