import math
a,b,x = map(int,input().split())
if (a*a*b)/2>=x :
    t = x*2/b/a/b
    ans = 90-math.degrees(math.atan(t))
else if (a*a*b)==x:
  ans = 0.0
else :
    x = (a*a*b) - x
    t = x*2/a/a
    ans = 90-math.degrees(math.atan(a/t))
print(ans)