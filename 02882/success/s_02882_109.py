import numpy as np
import math

a,b,x = map(int, input().split())
if 2 * x > a * a *b:
    ans = math.atan(2 * (a * a * b - x) / (a * a * a))
    ans = math.degrees(ans)
    print(ans)
else:
    ans = math.atan(a * b * b / (2 * x))
    ans = math.degrees(ans)
    print(ans)
    
