import math

a,h,x=map(int,input().split())

hanbun=x>(a*a*h/2)

if(hanbun):
    takasa=(x/a/a)*2-h
    katamuki=(h-takasa)/a
    print(math.degrees(math.atan(katamuki)))
else:
    haba=2*x/a/h
    print(math.degrees(math.atan(h/haba)))