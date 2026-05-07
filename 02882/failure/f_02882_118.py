a, b, x = map(int,input().split())
s = x/a
import math
if (a*b / 2 >= s):
  h = 2*s/b
  rad = math.atan(h/b)
  print(90.0 - math.degrees(rad))
else:
  rest = a*b - s
  l = 2*rest/a
  rad = math.atan(a/l)
  print(90.0 - math.degrees(rad))