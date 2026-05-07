import math
a,b,x=map(int,input().split())
print(math.atan(a*b*b/2/x*(x<a*a*b/2)or(b/a-x/a**3)*2)/565.486)