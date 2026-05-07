import math
a,b,x = map(int, input().split())
if a * a * b == x:
  print(0)
  exit()
if x / a / a >= b / 2:
  print(90 - math.degrees(math.atan(a/2/(b-(x/a/a)))))
else:
  print(90 - math.degrees(math.atan(2*x/a/b/b)))