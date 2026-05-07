import math
import numpy as np
a,b,x = map(int,input().split())

if x <= (a*a*b)/2:
  c = (2*x)/(a*b)
  tan = b/c
  print(np.rad2deg(math.atan(tan)))

elif x > (a*a*b)/2:
  c = ((2*x)/(a*a)) - b
  tan = (b-c)/a
  print(np.rad2deg(math.atan(tan)))
  