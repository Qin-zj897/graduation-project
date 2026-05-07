import math
a, b, x = map(int, input().split())
bw = x / a**2
print(90*(b-bw)/b)