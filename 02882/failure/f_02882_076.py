import math
a, b, x = map(int, input().split())

ans = math.degrees(math.atan((b-x/a**2)/a*2))
print(ans)