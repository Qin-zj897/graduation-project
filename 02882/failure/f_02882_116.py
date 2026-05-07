a,b,x=map(int,input().split())
import math

def gets(a,b,c):
    flg=True
    tanc=math.tan(c)
    if tanc>b/a:
        flg=False
    
    if flg:
        ans=a*b-a*a*math.tan(c)/2
    else:
        ans=(b*b*math.tan(math.pi/2-c))
    return ans*a

now=45
haba=22.49

"""
for i in range(90):
    m=i-0.0001
    print(gets(a,b,math.radians(i)))
"""

for bis in range(1000):
    c=math.radians(now)
    #print(gets(a,b,c),now,haba)
    res=gets(a,b,c)
    if abs(res-x)<10**-15:
        print(now)
        break
    elif res<x:
        now=(now-haba)
    else:
        now=(now+haba)
    haba/=2
else:
    print(now)