import math

a,b,x = map(int,input().split())

q = a*b*b/2/x

print(math.degrees(math.atan(q)))