import math

class _const:
    def __init__(self):
        self.PI=math.acos(-1)
        self.convToDEG=180/self.PI
const=_const()


a,b,x=map(int,input().split(" "))

halfVolume=a*a*b/2
print(halfVolume)

if x>=halfVolume:
    print(const.convToDEG*math.atan(2*(a*a*b-x)/a**3))

else:
    h=x/(a*b)
    print(const.convToDEG*math.acos(2*h/b))

