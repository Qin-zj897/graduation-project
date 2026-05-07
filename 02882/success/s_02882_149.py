import math
a,b,x = map(int,input().split())
if x/(a*a) <= (b/2):
    ans = 2*x/(a*b*b)
    print(90-math.degrees(math.atan(ans)))
else:
    ans = (2*a*a*b-2*x)/(a*a*a)
    print(math.degrees(math.atan(ans)))