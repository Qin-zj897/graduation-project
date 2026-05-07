import numpy as np
a,b,x = map(int,input().split())
V = a * a * b
dV = V - x
#y = max(V,dV)
#h = dV / V * b * 2
#theta = np.arctan(a / h) * 360 / (2 * np.pi)

if x >= V / 2:
  h = dV / V * b * 2
  theta = np.arctan(a / h) * 360 / (2 * np.pi)
  print(90-theta)
else:
  h = x / V * a * 2
  theta = np.arctan(h / b) * 360 / (2 * np.pi)
  print(90-theta)