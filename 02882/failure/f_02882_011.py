import math

a, b, x = map(int, input().split())

y = b - (x / a**2)
z = (4 * y**2 + a**2)**0.5

cos = (a**2 - 4*y**2 - z**2) / (-4 * y * z)

theta = math.degrees(math.acos(cos))

print('{:.10f}'.format(90 - theta))