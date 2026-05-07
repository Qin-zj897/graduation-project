import math
a,b,x = map(int,input().split())

l1 = 2 * (b - x/(a * a))
l2 = (2 * x) / (a * b)
if x >= a * a * b / 2:
    print(math.degrees(math.atan2(l1,a)))

else:
        print(math.degrees(math.atan2(b,l2)))
