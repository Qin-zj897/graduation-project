import numpy as np
a, b, x = map(int,input().split())
if a * a * b * (1 / 2) >= x:   
    l = (2 * x) / (a * b)  
else:
    l = (2 * (a * a * b - x)) / (a * b)
print(180 / np.pi * np.arctan(b / l))