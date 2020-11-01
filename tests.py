from grad2 import Value

def test_ops():
  def net(a, b):
    c = (a + b).relu()
    return a * c**2

  a = Value(4)
  b = Value(3)
  f = net(a, b)
  f.backward()
  print(a)

  delta = 10**-5
  numerical_grad_a = (net(a+delta, b) - net(a, b)) / delta
  numerical_grad_b = (net(a, b+delta) - net(a, b)) / delta

  eps = 10**-3
  assert a.grad - numerical_grad_a.data < eps, "Calculated gradient isn't the same as numerical estimate"
  assert b.grad - numerical_grad_b.data < eps, "Calculated gradient isn't the same as numerical estimate"
