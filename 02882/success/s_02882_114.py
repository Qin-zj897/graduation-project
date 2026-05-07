import math
a, b, x = map(int, input().split())
if a * a * b == x:
  print(0.0)
elif (a * a * b) / 2 > x:
  t = 2 * x / (a * b * b)
  print(90.0 - math.degrees(math.atan(t)))
else:
  t = (a**3) / (2 * (a*a*b-x))
  print(90.0 - math.degrees(math.atan(t)))