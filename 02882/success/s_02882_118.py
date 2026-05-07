import numpy as np
import math

a, b, x = map(int, input().split())

h = x / (a * a)

h2 = (b - h) * 2

if h2 <= b:
    deg = math.degrees(math.atan(h2 / a))
else:
    h3 = ((x / a) / b) * 2
    deg = math.degrees(math.atan(b / h3))

print(deg)


"""

"""