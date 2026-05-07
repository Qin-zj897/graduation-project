import sys
import os
import math
a,b,x = map(int,input().split())
ｙ = (2*x-a*a*b)/a*a
print(math.degrees(math.atan2(a,b-y)))
