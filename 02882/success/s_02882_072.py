import math
a,b,x = map(int,input().split())
v = a*a*b

if x >= (v/2):
    right = x - b*(a**2)
    left = -(a**2)/2
    y = right/left

    alpha = math.degrees(math.atan(y/a))
else:
    right = x
    left = b*a/2
    y = right/left
    alpha = math.degrees(math.atan(b/y))
    
print(alpha)