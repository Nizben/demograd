import numpy as np


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


class Adam:
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        # Initialize first and second moment estimates
        self.m = {id(p): np.zeros_like(p.data) for p in parameters}
        self.v = {id(p): np.zeros_like(p.data) for p in parameters}

    def step(self):
        self.t += 1
        for param in self.parameters:
            if param.grad is not None:
                pid = id(param)
                self.m[pid] = self.beta1 * self.m[pid] + (1 - self.beta1) * param.grad
                self.v[pid] = self.beta2 * self.v[pid] + (1 - self.beta2) * (
                    param.grad**2
                )
                m_hat = self.m[pid] / (1 - self.beta1**self.t)
                v_hat = self.v[pid] / (1 - self.beta2**self.t)
                param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None
