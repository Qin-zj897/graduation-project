import math
k=list(map(int,input().split()))
a = int(k[0])
b = int(k[1])
x = int(k[2])

i = 2*x/(a*b**2)
print(90 - math.degrees(math.atan(i)))