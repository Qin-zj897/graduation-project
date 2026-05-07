import math
a,b,x = map(int,input().split(' '))
x = x/a
#print(x)
# type1
if x >= a*b/2:
    t = 2*x/a -b
    ans = math.atan2(b-t,a)
else:
    t=2*x/b
    ans = math.atan2(b,t)
print(math.degrees(ans) )