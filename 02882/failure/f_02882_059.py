from math import atan
from math import degrees

a,b,x = map(int,input().split())
if (a**2)*b/2<=x:
  print(degrees(atan(2/a*(b-x/a**2))))
else:
  print(degrees(atan(a*b**2/x)))