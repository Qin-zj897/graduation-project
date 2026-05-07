import math

a,b,x=input().split()
a,b,x=[int(a),int(b),int(x)]

c=(2*b*(a**2)-2*x)/(a**3)
print(math.degrees(math.atan(c)))