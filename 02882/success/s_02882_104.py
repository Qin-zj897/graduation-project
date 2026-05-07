import math

a, b, x = map(int, input().split())
result = math.atan(2 * ( a*a*b - x) / (a*a*a))
if a * math.tan(result) > b:
    result = math.pi / 2 - math.atan(2 * x / (a*b*b))

print(math.degrees(result))
