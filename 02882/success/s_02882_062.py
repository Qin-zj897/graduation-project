import math
a, b, x = map(int, input().split())

if a**2 * b <= 2 * x:
    h = (a**2 * b - x) / a**2 * 2
    theta = h / (a**2 + h**2)**0.5
    ans = math.degrees(math.asin(theta))
else:
    h = 2 * x / a / b
    theta = h / (b**2 + h**2)**0.5
    ans = math.degrees(math.acos(theta))

print(ans)