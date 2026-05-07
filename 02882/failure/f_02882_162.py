import math
import sys
#input = sys.stdin.readline
sys.setrecursionlimit(10**7)
a, b, x = map(int, input().split())

if a*a*b/2 < x:
    l = 2*b - 2*x/(a*a)
    print(90 - math.degrees(math.atan(a/l)))

elif a*a*b/2 == x:
    print(90 - math.degrees(math.atan(a/b)))

else:
    l = 2*x / (a*b)
    print(90 - math.degrees(math.atan(l/b)))
