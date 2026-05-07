import math
a, b, x = list(map(int, input().split()))

h = x / (a*a)

if h > b / 2:
    ans = math.degrees(math.atan((b-h) / (a/2)))
else:
    ans = math.degrees(math.atan(a*b*b/(2*x)))

print(ans)
