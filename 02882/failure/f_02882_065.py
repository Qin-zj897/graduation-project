import math

a,b,x = map(int, input().split())

h = x / a / b * 2
ans = math.atan(b / h) * 180.0 / math.pi
print(ans)
