import math
a, b, x = map(int, input().split())
if a*a*b * (1/2) < x:
  t = math.degrees(math.atan(2*(b/a - x/(a ** 3))))
  print(t)
else:
  t = math.degrees(math.atan(a * (b ** 2)/ (2 * x)))
  print(t)