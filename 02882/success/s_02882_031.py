import math

def h():
    a,b,x = map(int,input().split())

    xdash = x / a
    FULL = a * b

    if xdash == FULL:
        return 0
    elif FULL / 2 == xdash:
        return b/a
    elif FULL / 2 > xdash:
        return b*b/xdash/2
    else:
        return 2*(a*b-xdash)/a/a

print(math.degrees(math.atan(h())))