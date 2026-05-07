import math

a, b, x = map(int, input().split())
ans = 0
if a*a*b > 2*x:
    ans =180.0/np.pi*math.atan((a*b*b)/(2*x))
else:
    ans = 180.0/np.pi*math.atan((2*(a*a*b-x))/a**3)

print(ans)

