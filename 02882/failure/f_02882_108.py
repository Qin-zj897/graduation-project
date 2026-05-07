from math import degrees, atan2
a, b, x = map(int, input().split())
if (a / x) / (a * b) < 1 / 2:
    ans = degrees(atan2(b, (2 * a) / (x * b)))
else:
    ans = degrees(atan2(a, 2 * b - (2*x / a **2)))
print(ans)