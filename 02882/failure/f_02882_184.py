import math
a,b,x=map(int,input().split())
s=2*x/a/a
s_=s-b
print(90-360/(2*math.pi)*math.atan(a/(b-s_)))