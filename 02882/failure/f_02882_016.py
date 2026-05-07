import math

a,b,x = map(int,input().split())

ans = math.radians(90)-math.atan((2*x)/(a*b*b))

print(math.degrees(ans))