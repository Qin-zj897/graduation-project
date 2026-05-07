import math
[a, b, x] = list(map(int, input().split()))
if a > b:
  tan = 2 * (b / a - x / a**3)
elif a < b:
  tan = (a * b**2) / (2 * x)
else:
  limit = a**3 / 2
  if limit < x:
    tan = 2 * (1 - x / a**3)
  else:
    tan = a**3 / (2 * x)
ans = math.degrees(math.atan(tan))
print(ans)