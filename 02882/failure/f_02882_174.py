import math

a,b,x = map(int, input().split())

def over_flag(a,b,x):
    return math.atan( (3/2*b-3*x)/a )*180/math.pi
 
def below_flag(a,b,x):
    return (0.5-math.atan(3*x/a*b*b)/math.pi)*180

def main(a,b,x):
    if x>a**2*b:
        print(0)
        return 0
    if x>1/6*a**2*b:
        print(over_flag(a,b,x))
    else:
        print(below_flag(a,b,x))
    return 0
main(a,b,x)