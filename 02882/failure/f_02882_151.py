a,b,x = list(map(int,input().split()))

if x*2 > b:
    import math

    y = 0
    y = 2*x/(a**2)-b

    print(90-math.degrees(math.atan(a/(y-b))))
else:
    import math

    y = 0
    y = 2*x/(a*b)

    print(90-math.degrees(math.atan(y/b)))