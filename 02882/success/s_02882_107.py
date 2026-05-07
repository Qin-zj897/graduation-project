import numpy as np
from math import degrees
a, b, x = map(int, input().split())

if x >= a**2*b/2:
    print(degrees(np.arctan((2*b/a-2*x/a**3))))
else:
    print(90-degrees(np.arctan(2*x/(a*b**2))))
