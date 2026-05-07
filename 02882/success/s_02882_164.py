from math import *

def check(a,b,mid):
   # if mid>pi/2-ep:
    #    return 0
  #  aa=tanh(90-mid)*b
    if a*tan(mid)<=b:
        ans=a*a*b
        aa=a*tan(mid)*a*a
        ans-=(aa)/2.0
    else:
        
        ans=(a*b*b/tan(mid))/2.0
    return ans
        
a,b,x=map(int,input().split())

low=0
high=pi/2
ep=0.0000000001
while low<high and high-low>=ep:
    mid=(low+high)/2
   # print(mid)
    st=check(a,b,mid)
   # print(st)
    
    if st<x:
        high=mid
    else:
        low=mid
print(low/pi*180)
    