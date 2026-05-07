import math
a, b, x = map(int, input().split())

x /= a

if x/a == b/2:
    print(math.degrees(math.atan2(b, a)))
elif x/a < b/2:
    print(math.degrees(math.atan2(b, 2*x/b)))
else:
    print(math.degrees(math.atan2(2*(b-x/a), a)))
