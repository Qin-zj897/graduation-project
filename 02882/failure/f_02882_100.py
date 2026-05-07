# coding: utf-8
# Your code here!

import math

side, height, volume = [int(i) for i in input().split()]

fullvolume = side * side * volume
space_height = (fullvolume - volume) / (side * side)

if volume*2 > fullvolume:
    print(math.degrees(math.atan(space_height/side)))
else:
    print(math.degrees(math.atan(height / side - (space_height-height))))