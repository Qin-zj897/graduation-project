import math
a, b, x = map(int, input().split())

def capa(theta):
    theta = theta*math.pi/180
    if math.tan(theta) >= b/a:
        return a*a*(b - a*math.tan(theta)/2)
    else:
        return a*b*b/(2*math.tan(theta))

left = 0
right = 90
while right - left > 10**(- 7):
    mid = (left + right)/2
    if capa(mid) > x:
        left = mid
    else:
        right = mid
print((left + right)/2)