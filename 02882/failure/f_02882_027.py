import math

a,b,x = input().split()
a = int(a)
b = int(b)
x = int(x)

def main(a,b,x):
    if ((a**2*b)-x) != 0:
        y = (a ** 3)/(2*((a**2*b)-x))
        θ = math.atan(y)
        result = math.degrees(math.pi/2-θ)
        return result
    else:
        return 0

print(main(a,b,x))
