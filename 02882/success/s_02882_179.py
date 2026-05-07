import math
a, b, x = map(int, input().split())

volume = a**2*b

# 45度のとき
# a^2*b = 2*x
if volume > 2*x :
  side = 2*x/(a*b)
  print(math.degrees(math.atan(b/side)))
else :
  side = (2/(a**2))*(volume-x)
  print(math.degrees(math.atan(side/a)))