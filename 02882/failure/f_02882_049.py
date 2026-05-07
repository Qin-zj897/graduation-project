import math
a, b, x = map(int, input().split())
s = x / a
if s > a * b / 2:
  rad = math.atan(a ** 3 / (2 * a * a * b - 2 * x))
  print(90 - (rad * 180 / math.pi))
else:
  rad = math.atan(a * b * b / (2 * x))
  print(rad * 180 / math.pi)