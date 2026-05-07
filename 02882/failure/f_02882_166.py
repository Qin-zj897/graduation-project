import sys
sys.setrecursionlimit(int(1e6))

a, b, x = list(map(int, input().split()))

import math

m = a*a*b - x
n = m / (a*a)
ans_rad = math.atan(n / (a/2))
ans = ans_rad * 180 / math.pi
print(ans)
