import math


a, b, x = map(int, input().split())
t = x / a / a
maxim = b-t

if t < b/2:
    ans = 90 - math.degrees((math.atan(2 * x / a / b / b)))
else:
    ans = math.degrees(math.atan(2 * maxim / a))

print(ans)