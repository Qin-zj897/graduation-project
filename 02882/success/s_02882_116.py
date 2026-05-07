import math
a,b,x = map(int,input().split())

#l = bから引いた長さ、三角形の短辺・
#(b-l/2)*a*a = x
if a*a*b/2 < x:
    l =(b-x/(a*a))*2
    # print("M")
    # print(l)
    print(math.degrees(math.atan(l/a)))
#l = aから引いた長さ、三角形の短辺・
#(a-l)*a*b/2 = x
else:
    l =a-(2*x)/(a*b)
    # print("N")
    # print(l)
    print(90-math.degrees(math.atan((a-l)/b)))