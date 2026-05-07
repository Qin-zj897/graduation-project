import numpy as np

f = lambda a, b, x: np.rad2deg(np.arctan((a**2*b-x)*2)/a**3)

x = map(int, input().split())
print(f(*x))