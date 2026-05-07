import math

a, b, x = map(float, input().split())

if x > (a ** 2) * b / 2:
    tan_theta = 2 * ((a ** 2) * b - x) / (a ** 3)
else:
    tan_theta = a * (b ** 2) / (2 * x)

theta = math.degrees(math.atan(tan_theta))

print(theta)
