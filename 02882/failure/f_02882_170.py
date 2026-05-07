import numpy as np

a, b, x = map(int,input().split())
c = b-(x/a**2)
d = a*b

if c <= b//2:
    print(np.degrees(np.arctan2(b-(x / a**2),a/2)))

else:
    print(np.degrees(np.arctan2(x / d),b))