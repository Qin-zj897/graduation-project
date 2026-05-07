import math
a, b, c = map(int, input().split())

# 容器に入っている水が半分以下の場合に困る
w = [2]
if a*a*b > 2 * c:
    print(w[4])
    h = (2*c)/(a*a)
    print(math.degrees(math.acos(h/a)))
else:
    h = 2*(a*a*b-c)/(a**2)
    print(math.degrees(math.atan(h/a)))
