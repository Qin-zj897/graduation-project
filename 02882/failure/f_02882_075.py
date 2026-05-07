a, b, x = map(float, input().split(' '))

if x >= a**2*b:
    y = 2*b-(2*x/a**2)
    print(math.degrees(math.atan(y/a)))
else:
    y = 2*x/(a*b)
    print(math.degrees(math.atan(b/y)))