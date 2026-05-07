#n=int(input())
a,b,x=map(int,input().split())
#l=list(map(int,input().split()))
#l=[list(map(int,input().split())) for i in range(n)]

import math

s=x/a
if s<= a*b/s:
    c=2*s/b
    deg=math.degrees(math.atan(c/b))
    ans=90.0-deg
    print('{:.9f}'.format(ans))
else:
    s=a*b-s
    h=2*s/a
    deg=math.degrees(math.atan(a/h))
    ans=90.0-deg
    print('{:.9f}'.format(ans))
