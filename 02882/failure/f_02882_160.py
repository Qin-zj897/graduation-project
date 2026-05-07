import math

a,b,x = map(int,input().split())

print(90-math.degrees(math.atan(2*x/(a*b*b))))