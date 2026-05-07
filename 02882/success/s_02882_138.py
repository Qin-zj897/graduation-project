import math

a, b, x = map(int, input().split())


if a*a*b >= x*2:
  d = 2*x/a/b
  print(math.atan(b/d)*180/math.pi)
else:
  d = 2*x/a/a - b
  print(math.atan((b-d)/a)*180/math.pi)

