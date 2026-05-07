a, b, x = list(map(int, input().split()))
ma = a*a*b

import math
y = 2*(ma-x)/(a*a*a)
ans = math.degrees(math.atan(y))

if (ans > 45):
  y = a*b*b/(2 * x)
  ans = math.degrees(math.atan(y))
  print(ans)
else:
  print(ans)