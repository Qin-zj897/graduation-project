import math
a, b, x = map(int, input().split())

if x >  a * a * b / 2:
  x =  a * a * b - x
  t = x * 2 / a / a / a
else:
  t = b / x * b * a / 2
print (math.degrees(math.atan(t)))