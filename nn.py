import random
from grad import Value
from typing import List

class Neuron:
  def __init__(self, in_dim: int):
    self.weights = [Value(random.uniform(-1, 1)) for _ in range(in_dim)]

  def __call__(self, x):
    activation = sum(wi * xi for wi,xi in zip(self.weights, x))
    return activation.relu()

  def parameters(self):
    return self.weights

class Layer:
  def __init__(self, in_dim: int, out_dim: int):
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.neurons = [Neuron(in_dim) for _ in range(out_dim)]

  def __call__(self, x):
    out = [n(x) for n in self.neurons]
    return out

  def parameters(self):
    return [p for n in self.neurons for p in n.parameters()]

  def __repr__(self):
    return f'Layer\nInput dimension: {self.in_dim}\nOutput dimension: {self.out_dim}'

class MLP:
  def __init__(self, in_dim: int, out_dim: int, layer_sizes: List[int]):
    layer_dims: List[int] = [in_dim, *layer_sizes, out_dim]
    self.layers = [Layer(layer_dims[i], layer_dims[i+1])\
                   for i in range(len(layer_dims)-1)]

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]

  def zero_grad(self):
    for p in self.parameters():
      p.grad = 0

  # Predict
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)

    return x
