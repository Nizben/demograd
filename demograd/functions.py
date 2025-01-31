import numpy as np
from . import utils
from demograd.tensor_engine import Tensor

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
        add = Add()
        add.inputs = [a, b]
        out = Tensor(
            a.data + b.data,
            requires_grad=a.requires_grad or b.requires_grad,
            depends_on=[add],
        )
        if out.requires_grad:
            add.output = out
            out.set_grad_fn(add)
        return out

    def backward(self, grad_output):
        grad_a = broadcast_backward(grad_output, self.inputs[0].data.shape)
        grad_b = broadcast_backward(grad_output, self.inputs[1].data.shape)
        return grad_a, grad_b


class Sub(Function):
    @staticmethod
    def apply(a, b):

        a = a if isinstance(a, Tensor) else Tensor(np.array(a))
        b = b if isinstance(b, Tensor) else Tensor(np.array(b))
        sub = Sub()
        sub.inputs = [a, b]
        out = Tensor(
            a.data - b.data,
            requires_grad=a.requires_grad or b.requires_grad,
            depends_on=[sub],
        )
        if out.requires_grad:
            sub.output = out
            out.set_grad_fn(sub)
        return out

    def backward(self, grad_output):
        grad_a = broadcast_backward(grad_output, self.inputs[0].data.shape)
        grad_b = broadcast_backward(-grad_output, self.inputs[1].data.shape)
        return grad_a, grad_b


class Mul(Function):
    @staticmethod
    def apply(a, b):

        a = a if isinstance(a, Tensor) else Tensor(np.array(a))
        b = b if isinstance(b, Tensor) else Tensor(np.array(b))
        mul = Mul()
        mul.inputs = [a, b]
        out = Tensor(
            a.data * b.data,
            requires_grad=a.requires_grad or b.requires_grad,
            depends_on=[mul],
        )
        if out.requires_grad:
            mul.output = out
            out.set_grad_fn(mul)
        return out

    def backward(self, grad_output):
        grad_a = broadcast_backward(
            grad_output * self.inputs[1].data, self.inputs[0].data.shape
        )
        grad_b = broadcast_backward(
            grad_output * self.inputs[0].data, self.inputs[1].data.shape
        )
        return grad_a, grad_b


class Div(Function):
    @staticmethod
    def apply(a, b):

        a = a if isinstance(a, Tensor) else Tensor(np.array(a))
        b = b if isinstance(b, Tensor) else Tensor(np.array(b))
        div = Div()
        div.inputs = [a, b]
        out = Tensor(
            a.data / b.data,
            requires_grad=a.requires_grad or b.requires_grad,
            depends_on=[div],
        )
        if out.requires_grad:
            div.output = out
            out.set_grad_fn(div)
        return out

    def backward(self, grad_output):
        grad_a = broadcast_backward(
            grad_output / self.inputs[1].data, self.inputs[0].data.shape
        )
        grad_b = broadcast_backward(
            -grad_output * self.inputs[0].data / (self.inputs[1].data ** 2),
            self.inputs[1].data.shape,
        )
        return grad_a, grad_b
