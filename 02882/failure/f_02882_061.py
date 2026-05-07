import math

a, b, x = map(int, input().split())
volume = a ** 2 * b

if x >= volume / 2:
  c = (volume - x) / (a * a / 2)
  tan_theta = c / a
else:
  c = x / (a * b / 2)
  tan_theta = b / c
  print(tan_theta)
  
print(math.degrees(math.atan(tan_theta)))