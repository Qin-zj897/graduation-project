import math 

a,b,x=map(int, input().split())
t=(a**2*b-x)/a**2*2/a
print(math.atan(t))
