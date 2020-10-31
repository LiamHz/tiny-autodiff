from operations import *
from core import get_gradients

def print_partial_grad(grads, node, node_name):
  print(f'Partial derivative of parent with respect to {node_name}: {grads[node]}')

def test_ops():
  def net(a, b):
    c = a * b - a / b
    d = Exp(Cos(c))
    e = Log(c)
    return d + e

  a = Var(4)
  b = Var(3)
  grads = get_gradients(net(a, b))

  delta = Var(10**-5)
  numerical_grad_a = (net(a+delta, b) - net(a, b)) / delta
  numerical_grad_b = (net(a, b+delta) - net(a, b)) / delta

  eps = 10**-3
  assert grads[a] - numerical_grad_a.value < eps, "Calculated gradient isn't the same as numerical estimate"
  assert grads[b] - numerical_grad_b.value < eps, "Calculated gradient isn't the same as numerical estimate"
