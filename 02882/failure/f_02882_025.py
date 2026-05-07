# D
import math
a, b, x = map(int, input().split())

tan1 = (2/a)*(b - (x/(a**2)))
tan2 = (1/2)*(a*(b**2)/x)

actan = math.atan(tan1)
moto = (x/a**2)
gen = (a*math.tan(actan))
judge = (moto-gen) >= 0

if judge:
    tan = tan1
else:
    tan = tan2

actan = math.atan(tan)
degree = math.degrees(actan)
print(degree)