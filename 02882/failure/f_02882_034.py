import math

def solv(a, b, x):
    if x == a ** 2 * b:
        return 0
    elif x <= a**2 * b / 2:
        tanTheta = (2 * x) / (a * b ** 2)
    else:
        tanTheta = (a ** 3) / 2 * (a ** 2 * b - x)
    return 90 - math.degrees(math.atan(tanTheta))

a, b, x = map(int, input().split(" "))
print(solv(a, b, x))