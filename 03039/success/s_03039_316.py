  
    
N,M,K = map(int,input().split())


mod = 10 ** 9 + 7
MAX = N*M
fac = [0] * MAX
finv = [0] * MAX
inv = [0] * MAX

def cominit():
    fac[0] = 1
    fac[1] = 1
    finv[0] = 1
    finv[1] = 1
    inv[1] = 1
    for i in range(2,MAX):
        fac[i] = fac[i-1] * i % mod
        inv[i] = mod - inv[mod%i] * (mod//i) % mod
        finv[i] = finv[i-1] * inv[i] % mod



def cmb(n,r):
    if n < 0 or r < 0 or r > n:return 0
    if r > n/2: r = n-r        
    return fac[n] * (finv[r] * finv[n-r] % mod) % mod
    
cominit()
ans= 0
NCK = cmb(N*M-2,K-2) 
M2 = M**2 * NCK
N2 = N**2 * NCK
for i in range(1,N):
    ans += M2 * i * (N-i)
    ans %= mod
    
for i in range(1,M):
    ans += N2 * i * (M-i)
    ans %= mod
        
print(ans)