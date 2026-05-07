from math import tan, atan, degrees, pi
error = 10**(-7)

def calc(th):
    if th > atan(a / b):
        return a**2 * b - a**3 / (2*tan(th))
    else:
        return a * b**2 * tan(th) / 2


a,b,x = map(int, input().split())

left = 0.0
right = pi/2
while right - left > error:
    mid = (left + right) / 2
    if calc(mid) <= x: #まだこぼれてない
        left = mid
    else: #こぼれ始めている
        right = mid

ans = (left + right) / 2
print(degrees(ans))