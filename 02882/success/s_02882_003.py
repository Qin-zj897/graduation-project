import math
[a,b,X] = list(map(int,input().split()))
h = (X/a)/a
S = h*a
if S < a*b/2:
    c = 2*S/b
    output = math.degrees(math.atan(b/c))
else:
    c = 2*S/a - b
    output = math.degrees(math.atan((b-c)/a))
print(output)
