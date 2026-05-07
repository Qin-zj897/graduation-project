import math
from decimal import Decimal
a,b,x = list(map(int, input().split()))
ans = 0

#in: 傾き 底面 高さ
#out: 面積
def fb(tilt, a, b):
    #x=0 or y=0
    #y=tilt*x + g
    g=b-a*tilt
    y = g
    x = -g/tilt
    if x>0:
        #(a-a,0) (a-a,b) (x-a,0)
        return abs(0*0 - (x-a)*b)/2
    elif y>0:
        #(a,0) (a,b) (0,y) (0,0)
        return (b+y)*a/2
    else:
        #(a,0) (a,b) (0,0)
        return abs(a*b - a*0)/2
#in: 角度 out: y=axのa
def f(angle):
    a =  math.tan(math.radians(angle))
    return a




high = 90
low = 0
for i in range(1000):
    mid = (high + low) / 2
    m = fb(f(mid), a, b) * a
    if m>x:
        low = mid
    elif m<x:
        high = mid
    else:
        ans = mid
        break

print(mid)