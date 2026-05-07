import math

a,b,x = list(map(int,input().split()))
k = a*a*b - x

c = 3*k /a**2

#tan(x) = c / a
print(90*c/a)