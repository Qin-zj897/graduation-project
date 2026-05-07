import math
a,b,x = map(int,input().split())
ans = math.degrees(math.atan(2*x/a*b*b))
print(ans)