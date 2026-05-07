from math import atan, degrees
a, b, x = map(int, input().split())
print(degrees(atan(2 * (a ** 2 * b - x) / a ** 3)) if x >= a ** 2 * b / 2 else 90 - degrees(atan(x / a / b ** 2)))
