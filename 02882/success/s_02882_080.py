a, b, x = list(map(int, input().split()))
import math
# a = 3
# b = 1
# x = 8

vol = a * a * b

if x > vol / 2 :
    # print("水が台形")
    c = (2 * b) - (2 * x) / a ** 2 # 水でない三角形の底辺
    print(math.degrees(math.atan( c / a )))
else:
    # print("水が三角形")
    c = 2 * x / ( a * b ) # 水の三角形の底辺
    print(math.degrees(math.atan( b / c )))
