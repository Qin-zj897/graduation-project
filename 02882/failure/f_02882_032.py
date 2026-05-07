import math
a, b, x = map(int, input().split())
if a*a*b >= 2*x:
    print(math.degrees(math.atan2(b, 2*x/(a*a))))
else:
    print(math.degrees(math.atan2(2*b - 2*x/a/a, a)))