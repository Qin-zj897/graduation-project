import math as m

a, b, x = map(int, input().split())

x = float(x)
d_max = float(90)
d_min = float(0)
res = float(100)

if x >= a**2*b:
    print(0)
    exit()

if x < a*a*b/100:
    d_max = float(1)

while res > 10e-8:
    d_mid = (d_max+d_min)/2
    l1 = a*m.tan(d_mid/180*m.pi)
    if l1-b > 0:
        l2 = (l1-b)*m.tan((90-d_mid)/180*m.pi)
    else:
        l2 = 0
    v = a*(a*b - 1/2*a*l1 + 1/2*(l1-b)*l2)
    res = abs(x-v)
    if x - v < 0:
        d_min = d_mid
    else:
        d_max = d_mid
print(d_mid)