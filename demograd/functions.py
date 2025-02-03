import numpy as np
from demograd.utils import broadcast_backward

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
        from demograd.tensor_engine import Tensor
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
        from demograd.tensor_engine import Tensor
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
        from demograd.tensor_engine import Tensor
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
        from demograd.tensor_engine import Tensor
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

class Exp(Function):
    @staticmethod
    def apply(a):
        from demograd.tensor_engine import Tensor
        a = a if isinstance(a, Tensor) else Tensor(np.array(a))
        exp = Exp()
        exp.inputs = [a]
        out = Tensor(
            np.exp(a.data),
            requires_grad=a.requires_grad,
            depends_on=[exp],
        )
        if out.requires_grad:
            exp.output = out
            out.set_grad_fn(exp)
        return out

    def backward(self, grad_output):
        grad_a = broadcast_backward(
            grad_output * np.exp(self.inputs[0].data), self.inputs[0].data.shape
        )
        return grad_a

class Pow(Function):
    @staticmethod
    def apply(a, b):
        from demograd.tensor_engine import Tensor
        a = a if isinstance(a, Tensor) else Tensor(np.array(a))
        b = b if isinstance(b, Tensor) else Tensor(np.array(b))
        pow = Pow()
        pow.inputs = [a, b]
        out = Tensor(
            a.data ** b.data,
            requires_grad=a.requires_grad or b.requires_grad,
            depends_on=[pow],
        )
        if out.requires_grad:
            pow.output = out
            out.set_grad_fn(pow)
        return out

    def backward(self, grad_output):
        grad_a = broadcast_backward(
            grad_output * self.inputs[1].data * (self.inputs[0].data ** (self.inputs[1].data - 1)),
            self.inputs[0].data.shape
        )
        grad_b = broadcast_backward(
            grad_output * self.inputs[0].data ** self.inputs[1].data * np.log(self.inputs[0].data),
            self.inputs[1].data.shape
        )
        return grad_a, grad_b

class Log(Function):
    @staticmethod
    def apply(a):
        from demograd.tensor_engine import Tensor
        a = a if isinstance(a, Tensor) else Tensor(np.array(a))
        log = Log()
        log.inputs = [a]
        out = Tensor(
            np.log(a.data),
            requires_grad=a.requires_grad,
            depends_on=[log],
        )
        if out.requires_grad:
            log.output = out
            out.set_grad_fn(log)
        return out

    def backward(self, grad_output):
        grad_a = broadcast_backward(
            grad_output / self.inputs[0].data, self.inputs[0].data.shape
        )
        return grad_a

class Neg(Function):
    @staticmethod
    def apply(a):
        from demograd.tensor_engine import Tensor
        a = a if isinstance(a, Tensor) else Tensor(np.array(a))
        neg = Neg()
        neg.inputs = [a]
        out = Tensor(
            -a.data,
            requires_grad=a.requires_grad,
            depends_on=[neg],
        )
        if out.requires_grad:
            neg.output = out
            out.set_grad_fn(neg)
        return out

    def backward(self, grad_output):
        grad_a = broadcast_backward(
            -grad_output, self.inputs[0].data.shape
        )
        return grad_a