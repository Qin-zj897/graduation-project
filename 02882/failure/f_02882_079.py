import sys
import math
a, b, x = [float(i) for i in sys.stdin.readline().split()]
if x < a**2*b / 2:
    print(90.0 - math.degrees(math.atan(x / (a*b**2/2))))
else:
    print(90.0 - math.degrees(math.atan(a**3 / 2 / (a**2*b-x))))