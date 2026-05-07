import math

A,B,X = map(int,input().split())

fullBottle = A*A*B

if fullBottle >= X*2:
    C = 2*X/(A*B)
    ans = math.degrees(math.atan(C/B))
    print(90-ans)
else:
    C = 2*(B-X/(A**2))
    ans = math.degrees(math.atan(C/A))
    print(ans)