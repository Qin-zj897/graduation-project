from math import atan, pi

a, b, x = map(int, input().split())

h = x / a**2


border = a**2 * b / 2

if x < border:
    theta = atan(2 * a * b**2 / x) * 180 / pi
else:
    theta = atan(2 / a * (b - h)) * 180 / pi

print(theta)