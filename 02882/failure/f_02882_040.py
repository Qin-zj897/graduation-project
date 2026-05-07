#%%
import math 

a, b, x = map(int, input().split())
if x >= a*a*b/2:
    y = 2*(b-x/(a**2))
    ans = math.degrees(math.atan(a/y))
    print(90 - ans)
else:
    y = 2*(x/(a*b))
    ans = math.degrees(math.atan(y/b))
    print(90 - ans)