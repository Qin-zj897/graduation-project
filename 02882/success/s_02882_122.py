import math

nums = [int(i) for i in input().split()]
a = nums[0]
b = nums[1]
x = nums[2]
h = x/(a**2)

if(h>=b/2):
    print(math.degrees(math.atan((b-h)/(a/2))))

else:
    print(math.degrees(math.atan(b/(2*h*a/b))))
