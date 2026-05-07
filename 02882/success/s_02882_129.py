import math

a,b,x = map(int, input().split())
x /= a

if 2* x > a * b:
    temp = 2*(a*b - x)/(a**2)
else:
    temp = b**2/(2*x)

print(math.degrees(math.atan(temp)))