import numpy as np
from functions import Function
from tensor_engine import Tensor
from utils import broadcast_backward


class ReLU(Function):
    @staticmethod
    def apply(a):
        a = a if isinstance(a, Tensor) else Tensor(np.array(a))
        out_data = np.maximum(0, a.data)
        out = Tensor(out_data, requires_grad=a.requires_grad)
        if out.requires_grad:
            relu = ReLU()
            relu.inputs = [a]
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
        out_data = 1 / (1 + np.exp(-a.data))
        out = Tensor(out_data, requires_grad=a.requires_grad)
        if out.requires_grad:
            sigmoid = Sigmoid()
            sigmoid.inputs = [a]
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
        out_data = np.tanh(a.data)
        out = Tensor(out_data, requires_grad=a.requires_grad)
        if out.requires_grad:
            tanh = Tanh()
            tanh.inputs = [a]
            tanh.output = out
            out.set_grad_fn(tanh)
        return out

    def backward(self, grad_output):
        a = self.inputs[0]
        grad_a = grad_output * (1 - self.output.data**2)
        grad_a = broadcast_backward(grad_a, a.data.shape)
        return (grad_a,)
