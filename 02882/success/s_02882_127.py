import math

a, b, x = map(int, input().split())

area_base = a ** 2
volum_bottle = area_base * b

k = (2 * (volum_bottle - x)) / a ** 2

if k <= b:
  K_t = k / a
  angle = math.degrees(math.atan(K_t))
else:
  L_t = a * (b ** 2) / (2 * x)
  angle = math.degrees(math.atan(L_t))
  
print(round(angle, 10))