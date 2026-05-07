from decimal import *
import math
a,b,x=map(int,input().split())
height=Decimal(x)/Decimal(a*a)

y1=Decimal(2*a*height)/Decimal(b)   #これってbを固定した時の高さじゃ
if y1>a:#a*height=(b+y)*a/2<=>y=2height-b
    y1=b-(2*height-b)
    print(math.degrees(math.atan(y1/a)))
else:#a*height=b*y/2なるy<=>y=2*a*height/b
    print(math.degrees(math.atan(b/y1)))