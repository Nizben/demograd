import numpy as np
from tensor_engine import Tensor

class Function:
    @staticmethod
    def apply(*args, **kwargs):
        raise NotImplementedError
    def forward(self, *args):
        raise NotImplementedError
    def backward(self, grad_output):
        raise NotImplementedError

class Add(Function):
    @staticmethod
    def apply(a, b):
        a = a if isinstance(a, Tensor) else Tensor(np.array(a))
        b = b if isinstance(b, Tensor) else Tensor(np.array(b))
        out = Tensor(a.data + b.data, requires_grad = a.requires_grad or b.requires_grad)
        if out.requires_grad:
            add =Add()
            add.inputs = [a, b]
            add.output = out
            out.set_grad_fn(add)
        return out

    def forward(self, grad_output):
        grad_a = grad_output
        grad_b = grad_output
        # Handle broadcasting
        grad_a = broadcast_backward(grad_a, self.inputs[0].data.shape)
        grad_b = broadcast_backward(grad_b, self.inputs[1].data.shape)
        return grad_a, grad_b



class Sub(Function):
    @staticmethod
    def apply(a, b):
        a = a if isinstance(a, Tensor) else Tensor(np.array(a))
        b = b if isinstance(b, Tensor) else Tensor(np.array(b))
        out = Tensor(a.data - b.data, requires_grad = a.requires_grad or b.requires_grad)
        if out.requires_grad:
            sub = Sub()
            sub.inputs = [a, b]
            sub.output = out
            out.set_grad_fn(sub)
        return out

    def forward(self, grad_output):
        grad_a = grad_output
        grad_b = -grad_output
        grad_a = broadcast_backward(grad_a, self.inputs[0].data.shape)
        grad_b = broadcast_backward(grad_b, self.inputs[1].data.shape)
        return grad_a, grad_b


class Mul(Function):
    @staticmethod
    def apply(a, b):
        a = a if isinstance(a, Tensor) else Tensor(np.array(a))
        b = b if isinstance(b, Tensor) else Tensor(np.array(b))
        out = Tensor(a.data * b.data, requires_grad = a.requires_grad or b.requires_grad)
        if out.requires_grad:
            mul = Mul()
            mul.inputs = [a, b]
            mul.output = out
            out.set_grad_fn(mul)
        return out

    def forward(self, grad_output):
        grad_a = grad_output * b.data
        grad_b = grad_output * a.data
        grad_a = broadcast_backward(grad_a, a.data.shape)
        grad_b = broadcast_backward(grad_b, b.data.shape)
        return grad_a, grad_b