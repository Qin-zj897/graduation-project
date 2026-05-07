a,b,x=map(int,input().split())
import math
ans=0
if a*a*b/2>=x:
    h=2*x/(a*b)
    ans=math.atan(b/h)
else:
    h=2*(a*a*b-x)/(a*a)
    ans=math.atan(h/a)
    
print(ans*180/math.pi)