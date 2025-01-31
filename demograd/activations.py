import numpy as np
from demograd.tensor_engine import Tensor
from demograd.utils import broadcast_backward

class Function:
    @staticmethod
    def apply(*args, **kwargs):
        raise NotImplementedError

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

class ReLU(Function):
    @staticmethod
    def apply(a):
        a = a if isinstance(a, Tensor) else Tensor(np.array(a))
        relu = ReLU()
        relu.inputs = [a]
        out_data = np.maximum(0, a.data)
        out = Tensor(out_data, requires_grad=a.requires_grad, depends_on=[relu])
        if out.requires_grad:
            relu.output = out
            out.set_grad_fn(relu)
        return out

    def backward(self, grad_output):
        a = self.inputs[0]
        grad_a = grad_output.copy()
        grad_a[a.data <= 0] = 0
        grad_a = broadcast_backward(grad_a, a.data.shape)
        return (grad_a,)

class Sigmoid(Function):
    @staticmethod
    def apply(a):
        a = a if isinstance(a, Tensor) else Tensor(np.array(a))
        sigmoid = Sigmoid()
        sigmoid.inputs = [a]
        out_data = 1 / (1 + np.exp(-a.data))
        out = Tensor(out_data, requires_grad=a.requires_grad, depends_on=[sigmoid])
        if out.requires_grad:
            sigmoid.output = out
            out.set_grad_fn(sigmoid)
        return out

    def backward(self, grad_output):
        a = self.inputs[0]
        grad_a = grad_output * self.output.data * (1 - self.output.data)
        grad_a = broadcast_backward(grad_a, a.data.shape)
        return (grad_a,)


class Tanh(Function):
    @staticmethod
    def apply(a):
        a = a if isinstance(a, Tensor) else Tensor(np.array(a))
        tanh = Tanh()
        tanh.inputs = [a]
        out_data = np.tanh(a.data)
        out = Tensor(out_data, requires_grad=a.requires_grad, depends_on=[tanh])
        if out.requires_grad:
            tanh.output = out
            out.set_grad_fn(tanh)
        return out

    def backward(self, grad_output):
        a = self.inputs[0]
        grad_a = grad_output * (1 - self.output.data ** 2)
        grad_a = broadcast_backward(grad_a, a.data.shape)
        return (grad_a,)

class Softmax(Function):
    @staticmethod
    def apply(a):
        a = a if isinstance(a, Tensor) else Tensor(np.array(a))
        softmax = Softmax()
        softmax.inputs = [a]
        out_data = np.exp(a.data) / np.exp(a.data).sum(axis=-1, keepdims=True)
        out = Tensor(out_data, requires_grad=a.requires_grad, depends_on=[softmax])
        if out.requires_grad:
            softmax.output = out
            out.set_grad_fn(softmax)
        return out

    def backward(self, grad_output):
        a = self.inputs[0]
        grad_a = grad_output * self.output.data * (1 - self.output.data)
        grad_a = broadcast_backward(grad_a, a.data.shape)
        return (grad_a,)

