MOD = 10**9 + 7
N,M,K= map(int,input().split())

MOD = 10**9 + 7
num = 2000020
fact = [1] * (num+1)
ifact = [1] * (num+1)

for i in range(1,num+1):
    fact[i] = (fact[i-1] * i) % MOD
ifact[-1] = pow(fact[-1],MOD-2,MOD)
for i in range(1,num+1)[::-1]:
    ifact[i-1] = (ifact[i] * i)% MOD

memo = {}
def nCr2(n,r):
    if (n,r) in memo:
        return memo[(n,r)]
    else:
        memo[(n,r)] = (fact[n] * ifact[r] * ifact[n-r]) % MOD
        return memo[(n,r)]

def nCr(n,r):
    if r > n:
        return 0
    return (fact[n] * ifact[r] * ifact[n-r]) % MOD

ans = 0
ans1 = 0
ans2 = 0

for j in range(2,K+1):
    ans1 += M**2 * nCr(2*(M-1),j-2)* nCr((N-2)*M, K-j) 
    ans1 %= MOD
ans1 *= sum((i*(N-i) for i in range(1,N)))

for j in range(2,K+1):
    ans2 += N**2 * nCr(2*(N-1),j-2) * nCr((M-2)*N, K-j)
    ans2 %= MOD
ans2 *= sum((i*(M-i) for i in range(1,M)))

ans = (ans1 + ans2) % MOD
'''

for i in range(1,N):
    print(i)
    for j in range(2,K+1):
        for k in range(1,j):
            ans += i * (N-i) *k*(j-k)*nCr(M,k) * nCr(M,j-k)* nCr((N-2)*M, K-j) 
            ans %= MOD

for i in range(1,M):
    print(i)
    for j in range(2,min(K,2*N)+1):
        for k in range(1,j):
            ans += i * (M-i) * k* (j-k) * nCr(N,k) * nCr(N,j-k) * nCr((M-2)*N, K-j)
            ans %= MOD
'''

print(ans)
