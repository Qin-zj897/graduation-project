import math
a,b,x = map(float,input().split())
theta = math.degrees(math.atan((2. * (((a ** 2) * b) - x)) / (a ** 3)))
print(theta)