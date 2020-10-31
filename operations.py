import numpy as np

class Ops:
  """Override standard arithmetic operators"""
  def __add__(self, other):
    return Add(self, other)

  def __mul__(self, other):
    return Mul(self, other)

  def __sub__(self, other):
    return Add(self, Neg(other))

  def __truediv__(self, other):
    return Mul(self, Inv(other))

class Var(Ops):
  """A leaf node"""
  def __init__(self, value: float):
    self.value = value

class Add(Ops):
  def __init__(self, a, b):
    self.value = a.value + b.value
    # Each tuple is the node's child, and the local derivative
    self.grad = [(a, 1), (b, 1)]

class Mul(Ops):
  def __init__(self, a, b):
    self.value = a.value * b.value
    self.grad = [(a, b.value), (b, a.value)]

class Neg(Ops):
  def __init__(self, a):
    self.value = -1 * a.value
    self.grad = [(a, -1)]

class Inv(Ops):
  def __init__(self, a):
    self.value = 1 / a.value
    # TODO Why is the local derivative this?
    self.grad = [(a, -a.value ** -2)]

class Sin(Ops):
  def __init__(self, a):
    self.value = np.sin(a.value)
    self.grad = [(a, np.cos(a.value))]

class Cos(Ops):
  def __init__(self, a):
    self.value = np.cos(a.value)
    self.grad = [(a, -np.sin(a.value))]

class Exp(Ops):
  def __init__(self, a):
    self.value = np.exp(a.value)
    self.grad = [(a, self.value)]

class Log(Ops):
  def __init__(self, a):
    self.value = np.log(a.value)
    self.grad = [(a, 1.0 / a.value)]
