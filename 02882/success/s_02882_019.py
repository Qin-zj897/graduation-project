import math
a,b,x = map(int,input().split())
if x >= a*a*b / 2:
    flag =1
    l = b-(x*2/(a*a)-b)
    res = math.atan(l/a)*180/math.pi
else:
    flag = 2
    l = x*2/(a*b)
    res = math.atan(b/l)*180/math.pi
print(res)

