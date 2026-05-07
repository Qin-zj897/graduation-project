import sys,math
input = sys.stdin.readline
a,b,x = list(map(int,input().split()))
print(math.degrees(math.atan(2*(a**2*b-x)/a**3)))