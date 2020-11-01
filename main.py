import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from nn import MLP
from grad import Value
from tests import test_ops

def main():
  lr = 0.001
  input_size = 50
  output_size = 10
  n_iterations = 100

  # Both input and "ground truth" are random vectors
  x = np.random.random(input_size)
  y = np.random.random(output_size)

  # Randomly initialize neural network weights
  #weights = to_value(np.random.random((input_size, output_size)))
  nn = MLP(input_size, output_size, [5, 10, 20])
  print(nn.layers[0])

  losses = []
  for i in tqdm(range(100)):
    y_pred = nn(x)
    loss = np.sum((y - y_pred) * (y - y_pred))
    losses.append(loss.data)

    loss.backward()
    for p in nn.parameters():
      p.data -= lr * p.grad

    nn.zero_grad()

  plt.plot(losses)
  plt.ylabel('Loss')
  plt.xlabel('Iteration')
  plt.title('Multilayer perceptron fitting random noise')
  plt.show()

if __name__ == '__main__':
  main()
