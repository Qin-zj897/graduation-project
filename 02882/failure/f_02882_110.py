import math

a, b, x = [int(i) for i in input().split()]

tan_theta = 2.0 * (b / a - x / (a**3))
#print(tan_theta)
print(math.degrees(math.atan(tan_theta)))