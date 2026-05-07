import math
a, b, x = map(int, input().split())
m = a*a*b
aa = a-x / (a*a)
bb = x / (a*b)
if aa >= bb:
    print(90-math.degrees(math.atan(((x/a/b))*2/b)))
else:
    print(math.degrees(math.atan((b-(x/a/a))*2/a)))