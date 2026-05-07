def fact(n):
    a = 1
    for i in range(n):
        a *= i+1
    return a

def conb(m, n):
    return fact(n) // fact(m) // fact(n-m)

N, M, K = map(int, input().split())

sum = 0
c = conb(K-2, N*M-2)
for m in range(M):
    sum += c * (M-m) * (N**2) * m
    sum %= 1000000007
for n in range(N):
    sum += c * (N-n) * (M**2) * n
    sum %= 1000000007

print(sum)