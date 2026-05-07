import math
a, b, x = map(int, input().split())

# 空いている空間の体積
emp = a*a*b - x
# 空いた空間の三角柱の辺のうち、傾けて長さが変動する一辺の長さ
t_length = (emp * 2) / (a*a)

# 水が入った部分が三角柱型になる時は上の処理、空いた空間が三角柱型であるうちは下の処理、
if t_length > b:
    print(90 - math.degrees(math.atan2((x*2) / (a*a), a)))
else:
    print(math.degrees(math.atan2(t_length, a)))
