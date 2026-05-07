N,M,K=map(int,input().split())
MOD=10**9+7

def comb(n, k):
    m = 1
    if n < 2 * k:
        k = n - k
    for i in range(1, k + 1):
        m = (m * (n - i + 1) / i)%MOD    
    return m

ans_x=0
ans_y=0
for i in range(1,N):
  ans_x+=(i*(N-i))
for i in range(1,M):
  ans_y+=(i*(M-i))
ans_x*=(M**2)
ans_y*=(N**2)
ans=comb(N*M-2,K-2)%MOD
ans*=(ans_x+ans_y)
print(int(ans%MOD))

