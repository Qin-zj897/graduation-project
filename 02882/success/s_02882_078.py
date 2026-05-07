import math
a,b,x = map(int,input().split())
V = a**2*b
if x >= V//2:
  rest_V = V-x
  tan = 2*rest_V/(a**3)
  print(math.degrees(math.atan(tan)))
  
else:
  rest_V = x
  tan = a*(b**2)/(rest_V*2)
  print(math.degrees(math.atan(tan)))
