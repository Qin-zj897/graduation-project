a, b, x = map(int, input().split())

if x <= a**2*b/2:
  t = (2*x)/(a*b**2)
else:
  t = (a**3)/(2*(a**2*b-x))
#print(t)

import math
a = math.atan(t)
a = math.degrees(a)
print(90-a)
