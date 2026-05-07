import numpy as np


a, b, x = [int(str) for str in input().strip().split()]

if x < a * a * b / 2:
    print(np.rad2deg(np.arctan(b * b / x / 2)))
else:
    print(np.rad2deg(np.arctan(2 * (b - x / (a ** 2)) / a)))