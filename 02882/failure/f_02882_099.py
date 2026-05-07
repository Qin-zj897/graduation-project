from math import *
a,b,x=input().split()
a=float(a)
b=float(b)
x=float(x)
y = 2*(a*a*b-x)/(a*a*a)
ans=(atan(y)*180)/pi
print(ans)
