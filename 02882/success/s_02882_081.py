a, b, x = map(int, input().split())

from math import atan
from math import pi
if x > a*a*b/2:
    theta = atan(2 * (a*a*b - x) / (a**3))
else:
    theta = atan(a*b*b / (2*x))
print(theta * 180 / pi)