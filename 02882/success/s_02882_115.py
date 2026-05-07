from math import acos, degrees
a,h,x = (int(x) for x in input().split())
S = x / a
Ss = a*h - S
a_start = a-(2*Ss / h - a)
if a_start > a:
    h_start = 2 * Ss / a
    big = (h_start**2 + a**2) ** 0.5
    angle = degrees(acos(a/big)) 
    
else:
    hb = 2*S / ((h**2+a_start**2)**0.5)
    angle = degrees(acos(hb/h))
print(angle)