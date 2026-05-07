# -*- coding: utf-8 -*-

import math

a, b, x = map(int, input().split())

V = a**2 * b
V_2 = V / a

x_2 = x / a

if x_2 > V_2/2:
    c = 2*x/a**2 - b
    theta = math.atan(a/(b-c))
    deg = math.degrees(theta)
else:
    c = 2*x/(a*b)
    theta = math.atan(c/b)
    deg = math.degrees(theta)

print("{:.9}".format(90-deg))
