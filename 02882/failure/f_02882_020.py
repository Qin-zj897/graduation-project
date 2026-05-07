import math

a, b, x = map(int, input().split())

if a**2*b >= 2*x:
  l, r = 0, math.degrees(math.atan(a/b))
  while r-l > pow(10, -7):
    k = (r+l)/2
    if x <= a*b*b*math.tan(math.radians(k))/2:
      r = k
    else:
      l = k
  ans = 90-(r+l)/2
else:
  l, r = 0, 90-math.degrees(math.atan(b/a))
  while r-l > pow(10, -7):
    k = (r+l)/2
    if a**2*b-x <= a**3*math.tan(math.radians(k))/2:
      r = k
    else:
      l = k
  ans = (r+l)/2
print(ans)


