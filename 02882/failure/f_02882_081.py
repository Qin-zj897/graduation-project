import sys
import itertools
# import numpy as np
import time
import math
 
sys.setrecursionlimit(10 ** 7)
 
from collections import defaultdict
 
read = sys.stdin.buffer.read
readline = sys.stdin.buffer.readline
readlines = sys.stdin.buffer.readlines

import math

a, b, x = map(int, input().split())

v = a * a * b
s = x / a


if x < v / 2:
    # print(math.degrees(math.atan(2 * s / b / b)))
    print(90 - math.degrees(math.atan(2 * s / b / b)))
else:
    print(90 - math.degrees(math.atan(a / (b - (2 * s / a - b)))))

