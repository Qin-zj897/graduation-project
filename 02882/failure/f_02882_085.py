a,b,x = map(int,input().split())

import math
if x > a**2 * b/2:
  tan = a/(2*b-2*x/a**2)
  t = math.atan(tan)
else:
  tan = 2 * x/(a * b**2)
  t = math.atan(tan)
  
print(90 - math.degrees(t))