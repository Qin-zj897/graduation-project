import math
a,b,x=map(int,input().split())
print(math.degrees(math.atan(2/a/a/a*(a*a*b-x) if a*a*b<=x*2 else a*b*b/2/x)))