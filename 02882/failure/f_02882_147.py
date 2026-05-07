import math

a,b,x = map(int,input().split())

if a*a*b <= 2*x:
    print(90-math.degrees(math.atan(a**3/(2*(a*a*b-x)))))
else:
    print(90-math.degrees(math.atan(2*x/(a*b*b))))

