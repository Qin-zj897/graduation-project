N,M,K=map(int, input().split())
div=10**9+7
def ncr(n, r):
    res = 1
    for i in range(1, r+1):
        res = res*n//i
        n = n-1
    return res % div

result = 0
comb = ncr(N*M-2, K-2)
for d in range(1, N):
    result += d*comb*M*M*(N-d)

for d in range(1, M):
    result += d*comb*(M-d)*N*N

print(result % div)
