import math

a, b, x = map(int, input().split())

v = b*a*a
if v < 2*x:
    print( math.degrees( math.atan(2*(a*a*b-x)/(a*a*a)) ))
elif v == 2*x:
    print( math.degrees( math.atan( b/a ) ) )
else:
    print( math.degrees( math.atan( v/(2*x) ) ) )