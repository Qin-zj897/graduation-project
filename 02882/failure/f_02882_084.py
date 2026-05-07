import math
a, b, x = map(int, input().split())

a = 2 * x / (a * b)

print(math.degrees(math.atan2(b, a)))