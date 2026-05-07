a,b,x=map(int,input().split())
x/=a

import math

if x<=a*b/2:
    print(math.degrees(math.atan(b*b/2/x)))

else:
    print(math.degrees(math.atan(2*(a*b-x)/a/a)))
    
