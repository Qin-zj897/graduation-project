import math

a, b, x = map(int, input().split())

tan_n_2 = (a**2) * b / x
tan_n = math.sqrt(tan_n_2)
n = math.degrees(math.atan(tan_n))

print(n)