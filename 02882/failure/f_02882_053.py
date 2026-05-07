import math
a, b, x = map(int, input().split( ))
V = (a**2)*b

if x*2 >= V:
  print(math.degrees(math.atan(2*l/a**3)))
else:
  print(math.degrees(math.atan(V/2*x)))
  