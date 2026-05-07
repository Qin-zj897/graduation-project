import math
p = list(map(int, input().split()))
a = p[0]
b = p[1]
x = p[2]

if x <= (a**2)*b/2:
  print(math.degrees(math.atan( a*(b**2)/(2*x) )))
else:
  print(math.degrees(math.atan( (2*(a**2)*b-2*x)/(a**3) )))