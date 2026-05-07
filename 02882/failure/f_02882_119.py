import math
a,b,x=map(int,input().split())
s=(2*a*a*b-2*x)/a/a/a
t=a*b*b/2/x
ans_S=math.degrees(math.atan(s))
ans_T=math.degrees(math.atan(t))
if math.tan(math.radians(s)) => b/a:
    print(ans_S)
else:
    print(ans_T)