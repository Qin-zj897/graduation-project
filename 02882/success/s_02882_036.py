import math

a, b, x = map(int,input().split())
v = a*a*b

if x > v/2:
    y = 2*(b-(x/(a*a)))
    naname = math.sqrt((y*y) + (a*a))
    cosine = a/naname
    print(math.degrees(math.acos(cosine)))
elif x < v/2:
    y = 2*x/(b*a)
    naname = math.sqrt((y*y)+(b*b))
    cosine = b / naname
    print(90-math.degrees(math.acos(cosine)))
else:
    naname = math.sqrt((a*a)+(b*b))
    cosine = a/naname
    print(math.degrees(math.acos(cosine)))
