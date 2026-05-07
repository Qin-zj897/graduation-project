from math import atan, degrees
a, b, x = map(int, input().split())
tan_ans = (a * b * b) / (2 * x)
ans = degrees(atan(tan_ans))
print(ans)