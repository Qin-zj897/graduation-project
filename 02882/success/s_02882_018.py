import math
a,b,x = map(int,input().split())
if x/(a*a) == b:
    print(0)
elif(x/(a*a)>=b/2):
    print(90 - (math.atan(a/(2*(b -(x/(a*a)))))*180/math.pi))    
else:
     print(90 - ((math.atan(2*x/(a*b*b))*180/math.pi)))