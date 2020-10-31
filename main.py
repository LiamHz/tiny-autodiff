from collections import defaultdict

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

class Leaf(Ops):
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

def get_gradients(parent_node):
  # defaultdict will return a value of 0 if the key doesn't exist
  gradients = defaultdict(lambda : 0)

  stack = parent_node.grad.copy()
  while stack:
    node, local_grad = stack.pop()
    gradients[node] += local_grad
    if not isinstance(node, Leaf):
      for child_node, child_local_grad in node.grad:
        # Add node and gradient to stack
        # When moving along the edges of the graph,
        # local derivatives are multiplied
        stack.append((child_node, child_local_grad * local_grad))

  return gradients

def print_partial_grad(grads, node, node_name):
  print(f'Partial derivative of parent with respect to {node_name}: {grads[node]}')

def main():
  def net(a, b):
    c = a * b
    d = a - c
    return d + a

  a = Leaf(4)
  b = Leaf(3)
  grads = get_gradients(net(a, b))

  print_partial_grad(grads, a, 'a')
  print_partial_grad(grads, b, 'b')

  delta = Leaf(10**-5)
  numerical_grad_a = (net(a+delta, b) - net(a, b)) / delta
  numerical_grad_b = (net(a, b+delta) - net(a, b)) / delta
  print('Numerical estimate for a:', numerical_grad_a.value)
  print('Numerical estimate for b:', numerical_grad_b.value)

if __name__ == '__main__':
  main()
