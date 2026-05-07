import math
a,b,x=map(int,input().split())
t=a*b*b/2/x if 2*x<a*a*b else 2/a*(b-x/a/a)
print(math.degrees(math.atan(t)))
