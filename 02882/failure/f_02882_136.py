import math


a, b, x = map(int, input().split())
t = x / a / a
maxim = b-t

ans = math.degrees(math.atan(2*maxim/a))
print(ans)
