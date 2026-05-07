a,b,x = map(int, input().split())
if x < (b*a**2)/2:
    c = 2*x / a**2
    print(90 - math.degrees(math.atan(c/a)))
else:
    x2 = x - b*a**2
    c = 2*x2 / a**2
    print(-math.degrees(math.atan(c/a)))