import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set()
        self._op = ''

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def backward(self, grad=None):
        if not self.requires_grad:
            return

        if grad is None:
            if self.data.size == 1:
                grad = np.ones_like(self.data)
            else:
                raise RuntimeError("grad must be specified for non-0 tensor")

        self.grad = grad if self.grad is None else self.grad + grad

        topo = []
        visited = set()

        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for child in tensor._prev:
                    build_topo(child)
                topo.append(tensor)

        build_topo(self)

        for tensor in reversed(topo):
            tensor._backward()

    # Overloading operators
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad if self.grad is not None else out.grad
            if other.requires_grad:
                other.grad = other.grad + out.grad if other.grad is not None else out.grad

        out._backward = _backward
        out._prev = {self, other}
        out._op = 'add'
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = other.data * out.grad
                self.grad = self.grad + grad_self if self.grad is not None else grad_self
            if other.requires_grad:
                grad_other = self.data * out.grad
                other.grad = other.grad + grad_other if other.grad is not None else grad_other

        out._backward = _backward
        out._prev = {self, other}
        out._op = 'mul'
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = out.grad @ other.data.T
                self.grad = self.grad + grad_self if self.grad is not None else grad_self
            if other.requires_grad:
                grad_other = self.data.T @ out.grad
                other.grad = other.grad + grad_other if other.grad is not None else grad_other

        out._backward = _backward
        out._prev = {self, other}
        out._op = 'matmul'
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad)

        def _backward():
            grad = out.grad.copy()
            grad[self.data <= 0] = 0
            if self.requires_grad:
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        out._prev = {self}
        out._op = 'ReLU'
        return out

    def sum(self):
        out = Tensor(np.array(self.data.sum()), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = out.grad * np.ones_like(self.data)
                self.grad = self.grad + grad_self if self.grad is not None else grad_self

        out._backward = _backward
        out._prev = {self}
        out._op = 'sum'
        return out

    def mean(self):
        out = Tensor(np.array(self.data.mean()), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = out.grad * np.ones_like(self.data) / self.data.size
                self.grad = self.grad + grad_self if self.grad is not None else grad_self

        out._backward = _backward
        out._prev = {self}
        out._op = 'mean'
        return out
