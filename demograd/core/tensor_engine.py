import numpy as np

# 1. Tensor Class with Subtraction Support and Gradient Reduction for Broadcasting
class Tensor:
    def __init__(self, data, requires_grad=False):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)  # Ensure data is float32
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
                raise RuntimeError("grad must be specified for non-scalar tensors")

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

        def _get_reduce_axes(operand_shape, output_shape):
            # Align shapes by prepending 1s
            ndim = max(len(operand_shape), len(output_shape))
            operand_shape_aligned = (1,) * (ndim - len(operand_shape)) + operand_shape
            output_shape_aligned = (1,) * (ndim - len(output_shape)) + output_shape
            # Identify axes where operand was broadcasted
            axes = tuple(
                i for i, (s_operand, s_out) in enumerate(zip(operand_shape_aligned, output_shape_aligned))
                if s_operand == 1 and s_out > 1
            )
            return axes

        def _backward():
            if self.requires_grad:
                axes_self = _get_reduce_axes(self.data.shape, out.grad.shape)
                if axes_self:
                    grad_self = out.grad.sum(axis=axes_self)
                    # Reshape to original shape if necessary
                    grad_self = grad_self.reshape(self.data.shape)
                else:
                    grad_self = out.grad
                self.grad = self.grad + grad_self if self.grad is not None else grad_self

            if other.requires_grad:
                axes_other = _get_reduce_axes(other.data.shape, out.grad.shape)
                if axes_other:
                    grad_other = out.grad.sum(axis=axes_other)
                    grad_other = grad_other.reshape(other.data.shape)
                else:
                    grad_other = out.grad
                other.grad = other.grad + grad_other if other.grad is not None else grad_other

        out._backward = _backward
        out._prev = {self, other}
        out._op = 'add'
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = other.data * out.grad
                # Handle broadcasting for multiplication
                if self.data.shape != grad_self.shape:
                    grad_self = grad_self.sum(axis=tuple(i for i in range(grad_self.ndim) if self.data.shape[i] == 1))
                    grad_self = grad_self.reshape(self.data.shape)
                self.grad = self.grad + grad_self if self.grad is not None else grad_self
            if other.requires_grad:
                grad_other = self.data * out.grad
                # Handle broadcasting for multiplication
                if other.data.shape != grad_other.shape:
                    grad_other = grad_other.sum(axis=tuple(i for i in range(grad_other.ndim) if other.data.shape[i] == 1))
                    grad_other = grad_other.reshape(other.data.shape)
                other.grad = other.grad + grad_other if other.grad is not None else grad_other

        out._backward = _backward
        out._prev = {self, other}
        out._op = 'mul'
        return out

    def __rmul__(self, other):
        return self * other

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
        out = Tensor(np.array(self.data.sum(), dtype=np.float32), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = out.grad * np.ones_like(self.data)
                self.grad = self.grad + grad_self if self.grad is not None else grad_self

        out._backward = _backward
        out._prev = {self}
        out._op = 'sum'
        return out

    def mean(self):
        out = Tensor(np.array(self.data.mean(), dtype=np.float32), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = out.grad * np.ones_like(self.data) / self.data.size
                self.grad = self.grad + grad_self if self.grad is not None else grad_self

        out._backward = _backward
        out._prev = {self}
        out._op = 'mean'
        return out

    # Implementing subtraction
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other + (-self)

    # Implementing negation
    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = -out.grad
                self.grad = self.grad + grad_self if self.grad is not None else grad_self

        out._backward = _backward
        out._prev = {self}
        out._op = 'neg'
        return out

    # Implementing division (optional, if needed)
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other ** -1

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other * self ** -1

    # Implementing power (optional, if needed)
    def __pow__(self, power):
        assert isinstance(power, (int, float)), "Only supporting int/float powers for now."
        out = Tensor(self.data ** power, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = power * (self.data ** (power - 1)) * out.grad
                self.grad = self.grad + grad_self if self.grad is not None else grad_self

        out._backward = _backward
        out._prev = {self}
        out._op = f'pow_{power}'
        return out

# 2. Optimizers
class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
            if param.grad is not None:
                param.data -= self.lr * param.grad

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None

# 3. Neural Network Modules
class Linear:
    def __init__(self, in_features, out_features):
        # Initialize weights with small random numbers
        self.weight = Tensor(np.random.randn(in_features, out_features).astype(np.float32) * 0.01, requires_grad=True)
        self.bias = Tensor(np.zeros(out_features, dtype=np.float32), requires_grad=True)

    def __call__(self, x):
        return x @ self.weight + self.bias

    def parameters(self):
        return [self.weight, self.bias]

class ReLU:
    def __call__(self, x):
        return x.relu()

class MSELoss:
    def __call__(self, pred, target):
        loss = (pred - target) * (pred - target)
        return loss.mean()  # Use the mean method to compute average loss

# 4. Neural Network Definition
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.fc1 = Linear(input_size, hidden_size)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size, output_size)

    def __call__(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()

# 5. Training Loop with Dummy Data
if __name__ == "__main__":
    # Create dummy data
    np.random.seed(42)
    X = Tensor(np.random.randn(10, 3).astype(np.float32), requires_grad=False)  # 10 samples, 3 features
    y = Tensor(np.random.randn(10, 2).astype(np.float32), requires_grad=False)  # 10 samples, 2 targets

    # Initialize the network, loss, and optimizer
    model = SimpleNN(input_size=3, hidden_size=5, output_size=2)
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.1)

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        # Forward pass
        pred = model(X)
        loss = criterion(pred, y)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Zero gradients
        optimizer.zero_grad()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.data}")

    # Final loss
    print(f"Final Loss: {loss.data}")
