import math
a,b,x = map(float,input().split())
w_theta = math.atan(b / a)
w_p = 0.5 * a * b * math.sin(w_theta) * math.sqrt(a ** 2 + b ** 2)
if x > w_p:
  theta = math.degrees(math.atan((2. * (((a ** 2) * b) - x)) / (a ** 3)))
  else:
    theta = math.degrees(math.atan((a * (b ** 2))/ (2. * x)))
    print(theta)