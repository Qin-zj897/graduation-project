import math
Input = list(map(int,input().split()))
a = Input[0]
b = Input[1]
x = Input[2]

taiseki = a**2*b

if x>taiseki/2:
    theta = -math.atan(x/(a**2*(b-a)))
else:
    theta = math.atan(x/(a**2*b))
print(theta)
