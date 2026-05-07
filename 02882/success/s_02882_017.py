import math

a, b, x = map(int, input().split())
v = a * a * b
if(x <= v / 2):
    h = 2 * x / a / b
    ans = math.degrees(math.atan(b / h))
else:
    h = (v - x) / a / a * 2
    ans = math.degrees(math.atan(h / a))

print(ans)