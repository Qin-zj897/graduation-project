import math
a, b, x = map(int, input().split())
h = x * 2 / b / a
if h <= a:
    rad = math.atan2(b, h)
    print(math.degrees(rad))
else:
    emp_h = (a*a*b - x)*2 / a / a
    rad = math.atan2(a, emp_h)
    print(90 - math.degrees(rad))
