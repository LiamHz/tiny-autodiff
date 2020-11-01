import numpy as np
from collections import defaultdict

class Value:
  def __init__(self, data, children=()):
    self.data = data
    self._backward = lambda : None

    self.prev = set(children)
    self.grad = 0

  def __add__(self, other):
    # Handle scalar operators
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other))

    # Called during reverse mode differentiation
    def _backward():
      self.grad += 1 * out.grad
      other.grad += 1 * out.grad

    out._backward = _backward
    return out

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other))

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad

    out._backward = _backward
    return out

  def __pow__(self, other):
    out = Value(self.data ** other, (self,))

    def _backward():
      self.grad += (other * self.data**(other-1)) * out.grad

    out._backward = _backward
    return out

  def relu(self):
    out = Value(self.data if self.data > 0 else 0, (self,))

    def _backward():
      self.grad += (out.data > 0) * out.grad

    out._backward = _backward
    return out

  def __truediv__(self, other):
    return self * other**-1

  def __neg__(self):
    return self * -1

  def __sub__(self, other):
    return self + (-other)

  def __radd__(self, other):
      return self + other

  def __rsub__(self, other):
      return other + (-self)

  def __rmul__(self, other):
      return self * other

  def __truediv__(self, other):
      return self * other**-1

  def __rtruediv__(self, other):
      return other * self**-1

  def backward(self):
    visited = set()
    graph = []

    # Recursively build autodiff graph
    def build_graph(node):
      if node not in visited:
        visited.add(node)
        for child in node.prev:
          build_graph(child)
        graph.append(node)

    build_graph(self)

    self.grad = 1

    # TODO Why is this reversed?
    for node in reversed(graph):
      node._backward()

  def __repr__(self):
    return f"Value(data={self.data}, grad={self.grad})"
