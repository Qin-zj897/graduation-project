import math
a,b,x = [int(x) for x in input().split()]
c = 2*b/a - 2*x/a**3
if c > 2*x/a**3:
  print(math.degrees(math.acos(x/(a**2*b))))
else:
  print(math.degrees(math.acos(x/(a**2*b))))