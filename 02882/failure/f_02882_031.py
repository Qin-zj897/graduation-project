import math
a, b, x = map(int, input().split())
_x = x / a
if _x == a * b:
    print(90)
elif _x <= a * b / 2:
    theta = math.atan(_x * 2 / b / b)
    print(90 - math.degrees(theta))
else:
    _x = a * b - _x
    theta = math.atan(a * a / 2 / _x)
    print(90 - math.degrees(theta))