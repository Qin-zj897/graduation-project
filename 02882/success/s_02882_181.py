import math

a, b, x = map(int, input().split())
if x > a**2*b/2:
  degree = math.degrees(math.atan2(2*a**2*b-2*x, a**3))
  print(degree)
else:
  degree = math.atan2(2*x, a*(b**2))
  degree = math.pi/2 - degree
  print(math.degrees(degree))