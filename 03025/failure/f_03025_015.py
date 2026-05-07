import math

N,A,B,C = map(int, input().split())
A=A/100
B=B/100
ans = 0

def combinations_count(n, r):
    return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))


for i in range(N):
    ans=ans+A**(N-1)*(1-A)**i*combinations_count(N, i)
    ans=ans+A**(N-1)*(1-B)**i*combinations_count(N, i)

print(ans)
