#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import pi, atan2

a, b, x = [int(x) for x in input().split()]
# a, b, x = [2, 2, 4]
# a, b, x = [12, 21, 10]
# a, b, x = [3, 1, 8]

s = x / a

if s < (a * b) / 2:
    # w = (2 * s) / b コース
    w = (2 * s) / b
    rad = atan2(b, w)
else:
    # h = ((a * b - s) * 2) / a コース
    h = ((a * b - s) * 2) / a
    rad = atan2(h, a)

# radian から角度への変換 [radian に (180 / pi) をかける]
print(rad * (180 / pi))
