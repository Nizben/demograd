from tensor_engine import Tensor

a = Tensor([2.0], requires_grad=True)
b = Tensor([3.0], requires_grad=True)
c = a + b * b
c.backward()
print(c._grad_fn)
print(a.grad)  # Output: [1.0]
print(b.grad)  # Output: [1.0]
