a,b,x=map(int, input().split())

import math
if 0.5*a**2*b<=x:
    ans=math.atan(2*(1/(a**2)*(a*b-x/a)))
    ans=math.degrees(ans)
else:
    ans=math.atan((a*b**2)/(x/2))
    ans=math.degrees(ans)

print(ans)