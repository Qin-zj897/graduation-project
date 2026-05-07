import math
a,b,x = list(map(int,input().split()))



theta = math.degrees(math.atan(b*b*a/(2*x)))
if a*a*b <=  x-1e-9:
    print(0)
elif a*a*b/2 >= x:
    print(theta)
else:
    L = 1e-6
    R = theta
    while R-L > 1e-7:
        theta = (R+L)/2
        beta = math.degrees(math.atan(b/a))
        alpha = beta-theta
        xp = a*(a*a+b*b)*math.sin(math.radians(alpha))*math.cos(math.radians(alpha))/2 - a*(a*a+b*b)*math.sin(math.radians(alpha))*math.sin(math.radians(alpha))/(2*math.tan(math.radians(90-beta+alpha)))  + a*a*b/2
        if x< xp:
            L = theta
        else:
            R = theta

    print(R)
