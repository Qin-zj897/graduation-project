import math

a, b, x = map(int, input().split())

if a**2*b >= 2*x:
  l, r = 0, 45
  while r-l > pow(10, -14):
    k = (r+l)/2
    if x <= a*b*b*math.atan(math.radians(k))/2:
      r = k
    else:
      l = k
  ans = 90-(r+l)/2
else:
  l, r = 0, 45
  while r-l > pow(10, -14):
    k = (r+l)/2
    if a**2*b-x <= a**3*math.tan(math.radians(k))/2:
      r = k
    else:
      l = k
  ans = (r+l)/2
print(ans)