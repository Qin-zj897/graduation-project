import math

a, b, x = map(int, input().split())
if a * a * b <= 2 * x:
    ans = math.degrees(math.atan((2 * (b - (x / (a * a))) / a)))
    print('{:.10f}'.format(ans))
else:
    ans = math.degrees(math.atan((a * b * b) / (2 * x)))
    print('{:.10f}'.format(ans))