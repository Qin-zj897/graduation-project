# D
import math
A, B, C = map(int, input().split())
x = C / A**2
height = 2 * x - B
if (height >= 0):
    y = B -height
    print(math.degrees(math.atan(y/A)))
else:
    z = 2*C / (A*B)
    print(math.degrees(math.atan(B/z)))