import math
a, b, x = map(int, input().split())

if a ** 2 * b >= x * 2:
  print(math.degrees(math.atan(b ** 2 * a / (x * 2))))
else:
  r = a ** 2 * b - x
  print(math.degrees(math.atan(r * 2 / a ** 3)))