import math

a,b,x = list(map(int, input().strip().split()))
h = x/a/a
if h < b/2:
	aa = 2*x/a/b
	ans = math.atan2(aa, b) * 180 / math.pi
else:
	bb = 2*(a*a*b - x)/a/a
	ans = math.atan2(a, bb) * 180 / math.pi
print(90 - ans)