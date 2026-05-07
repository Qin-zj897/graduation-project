import math
a,b,x = [int(x) for x in input().split()]
c = 2*b/a - 2*x/a**3
if c > 2*x/a**3:
  print(math.degrees(math.atan(a*b**2/(2*x))))
else:
  print(math.degrees(math.atan(c)))