from math import pi
import numpy as np
a, b, x = map(int, input().split())

ans_rad = np.arctan(2 * x / (a * b**2))
ans = 90 - ans_rad * (180 / pi)
print("{}".format(ans))