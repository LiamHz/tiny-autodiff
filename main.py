from tests import test_ops
import numpy as np
from grad import Value, get_gradients
import matplotlib.pyplot as plt

def update_weights(weights, grads, lr):
  for i in range(weights.shape[0]):
    for j in range(weights.shape[1]):
      weights[i, j].value -= lr * grads[weights[i, j]]

def main():
  lr = 0.01
  min_pct_delta_loss = 0.001
  min_iters = 3
  input_size = 50
  output_size = 10

  # Functions to vectorize and un-vectorize Value objects 
  to_var = np.vectorize(lambda x: Value(x))
  to_vals = np.vectorize(lambda var: var.value)

  # Both input and "ground truth" are random vectors
  x = to_var(np.random.random(input_size))
  y = to_var(np.random.random(output_size))

  # Randomly initialize neural network weights
  weights = to_var(np.random.random((input_size, output_size)))

  losses = []
  for i in range(1000):
    y_pred = np.dot(x, weights)

    loss = np.sum((y - y_pred) * (y - y_pred))
    losses.append(loss.value)

    # Stop training early if loss isn't changing much
    try:
      pct_delta_loss = (losses[-2] - losses[-1]) / losses[-2]
    except IndexError:
      pct_delta_loss = 0.1
    if pct_delta_loss < min_pct_delta_loss:
      break

    grads = get_gradients(loss)
    update_weights(weights, grads, lr)

  plt.plot(losses)
  plt.ylabel('Loss')
  plt.xlabel('Iteration')
  plt.title('Single layer network fitting random noise')
  plt.show()

if __name__ == '__main__':
  main()
