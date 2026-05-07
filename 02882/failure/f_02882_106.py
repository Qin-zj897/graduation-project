from math import degrees, atan
a, b, c = map(int, input().split())

h = c/a**2
print(degrees(atan(2*(b-h)/a)))