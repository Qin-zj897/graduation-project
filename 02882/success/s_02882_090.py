import math
a,b,x=map(int,input().split())
if a*a*b/2 == x:
    # 丁度半分
    h=b
    w=a
elif a*a*b/2 > x:
    # 半分未満
    h=b
    w=2*x/b/a
else:
    # 半分越え
    h=2*(b-x/a/a)
    w=a
print(math.degrees(math.atan(h/w)))