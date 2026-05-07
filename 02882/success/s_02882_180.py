from math import atan,atan2,degrees

a, b ,x = map(int ,input().split())

# 半分より多いとき
if a * a * b / 2 < x:
  print(degrees(atan(2*b/a -2*x/(a*a*a))))
else:
  # 半分より少ないとき
  print(90-degrees(atan(2*x/(a*b*b))))
