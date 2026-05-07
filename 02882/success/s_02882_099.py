a,b,v = map(int,input().split())
import math
s = v/a
if 2*s < a*b:
  sin = b/math.sqrt((b**2)+(4*(s**2)/(b**2)))
  print(math.degrees(math.asin(sin)))
elif 2*s == a*b:
  sin = b/math.sqrt((b**2)+(a**2))
  print(math.degrees(math.asin(sin)))
else:
  sin = 2*(a*b-s)/math.sqrt(((4*((a*b-s)**2))/(a**2))+(a**2))/a
  print(math.degrees(math.asin(sin)))
  