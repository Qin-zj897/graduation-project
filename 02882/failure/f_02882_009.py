import math
a,b,x = map(float,input().split())

c = a/(2*b-2*x/a**2)
d = (2*x)/(a*b**2)
if x/a >= a*b/2:
  result = math.atan(c)
else:
  result = math.degrees(math.atan(d))
print(result)
  