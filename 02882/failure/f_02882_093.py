import math
a, b, x = map(int, input().split())
if x <= a*a*b/2:
    low, high = 45.0, 90.0
    while high - low > 0.0000000001:
        key = (low+high)/2
        if math.tan(math.radians(90-key))*b*b*a/2 > x:
            low = key
        else:
            high = key
    print(key)
else:
    v = a*a*b
    low, high = 0.0, 45.0
    while high - low > 0.0000000001:
        key = (low+high)/2
        if v - math.tan(math.radians(key))*a*a*a/2 > x:
            low = key
        else:
            high = key
    print(key)