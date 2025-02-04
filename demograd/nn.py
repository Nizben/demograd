import numpy as np
from demograd.tensor_engine import Tensor
from demograd.functions import MatMul


class Linear:
    def __init__(self, in_features, out_features, bias=True):
        # Initialize weights and bias
        self.W = Tensor(
            np.random.randn(in_features, out_features) * 0.01,
            requires_grad=True,
            name="W",
        )
        if bias:
            self.b = Tensor(np.zeros((1, out_features)), requires_grad=True, name="b")
        else:
            self.b = None
        self.parameters = [self.W] + ([self.b] if self.b is not None else [])

    def forward(self, x):
        # x is expected to be a Tensor of shape (batch_size, in_features)
        # Compute x @ W (using our MatMul) and add bias
        out = MatMul.apply(x, self.W)
        if self.b is not None:
            out = out + self.b  # Broadcasting over the batch dimension
        return out

    def __call__(self, x):
        return self.forward(x)


class Sequential:
    def __init__(self, *layers):
        self.layers = layers
        self.parameters = []
        for layer in layers:
            if hasattr(layer, "parameters"):
                self.parameters += layer.parameters

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __call__(self, x):
        return self.forward(x)
