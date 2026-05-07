import math
a, b, x = list(map(int,input().split()))
h = x / (a * b * 0.5)
if h > a:
    u = (a ** 2 * b - x) / (a ** 2 / 2)
    ans = math.degrees(math.atan2(u, a))
else:
    ans = math.degrees(math.atan2(b, h))
print(ans)