a, b, x = map(int, input().split())
v = a**2 * b
import math
if x >= v/2:
    ans = math.degrees(math.atan(2 * (v - x) / a**3))
else:
    ans = math.degrees(math.atan(a * b**2 / (2 * x)))
print(ans)