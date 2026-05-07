import math
a, b, x = map(int, input().split())
if a**2*b/2 > x:
  print('case1')
  ans = math.atan(float(a*b*b / (2*x))) * 180 / math.pi
else:
  ans = math.atan(2*b/a -2*x/a**3) * 180 / math.pi
print(ans)