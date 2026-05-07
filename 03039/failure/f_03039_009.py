N,M,K=map(int,input().split())
MOD=10**9+7
def comb(n, k):
    m = 1
    if n < 2 * k:
        k = n - k
    for i in range(1, k + 1):
        m = m * (n - i + 1) / i    
    return m
for i in range(1,N):
  ans_x+=(i*(N-i))
for i in range(1,M):
  ans_y+=(i*(M-i))
ans_x*=M**2
ans_y*=X**2
print(conb(NM-2,K-2)*(ans_x+ans_y)%MOD)
