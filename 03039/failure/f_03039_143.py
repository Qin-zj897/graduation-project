from math import factorial

def comb(n, r):
    return factorial(n) // (factorial(n - r) * factorial(r))

N,M,K = map(int,input().split())

C = comb(N*M-2, K-2)
print(int(C))

A = int(0)
for d in range(N):
    a = (N-d)*(M**2)

    A += int(d*a)

B = int(0)
for d in range(M):
    b = (M-d)*(N**2)
    B += int(d*b)

ans = int((A+B)*C)
print(ans)
