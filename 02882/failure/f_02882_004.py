import math
import sys
ans=float(0)
a, b, x =map(float, input().split())

if x==a*a*b:
    ans=0
    sys.exit()
    ans=math.degrees(ans)
    ans=90-ans
    print('{:.7f}'.format(ans))

c=(2*x/a/a)-b
if c > 0:
    ans=math.atan(a/(b-c))

else:
    ans=math.atan(2*x/a/b/b)
ans=math.degrees(ans)
ans=90-ans
print('{:.7f}'.format(ans))
