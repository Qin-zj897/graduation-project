import math
a,b,x = (int(i) for i in input().split())

ans = math.degrees(math.atan(2*(a**2*b-x)/a**3))
print(ans)