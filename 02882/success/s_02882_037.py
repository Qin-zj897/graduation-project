# coding: utf-8
# Your code here!

import math

side, height, volume = [int(i) for i in input().split()]

fullvolume = side * side * height
space_height = (fullvolume - volume) / (side * side)

if volume*2 >= fullvolume:
    print(math.degrees(math.atan(space_height*2 / side)))
else:
    print(math.degrees(math.atan(height / (2*volume/side/height) )))