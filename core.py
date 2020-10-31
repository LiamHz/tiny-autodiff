from collections import defaultdict
from operations import *

def get_gradients(parent_node):
  # defaultdict will return a value of 0 if the key doesn't exist
  gradients = defaultdict(lambda : 0)

  stack = parent_node.grad.copy()
  while stack:
    node, local_grad = stack.pop()
    gradients[node] += local_grad
    if not isinstance(node, Var):
      for child_node, child_local_grad in node.grad:
        # Add node and gradient to stack
        # When moving along the edges of the graph,
        # local derivatives are multiplied
        stack.append((child_node, child_local_grad * local_grad))

  return gradients

