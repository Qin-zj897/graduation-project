n,m,k = map(int, input().split())

#############

mod = 10**9 + 7
max = n*m + 10
fac, finv, inv = [0]*max, [0]*max, [0]*max

def comInit(max):
    fac[0] = fac[1] = 1
    finv[0] = finv[1] = 1
    inv[1] = 1

    for i in range(2,max):
      fac[i] = fac[i-1]* i% mod
      inv[i] = mod - inv[mod%i] * (mod // i) % mod
      finv[i] = finv[i-1] * inv[i] % mod

comInit(max)

# 二項係数の計算
def com(n,k):
    if(n < k):
        return 0
    if( n<0 | k < 0):
        return 0
    return fac[n] * (finv[k] * finv[n-k] % mod) % mod

#############

ans = n*n*(m-1)*m*(m+1)//6 + m*m*(n-1)*n*(n+1)//6
ans = ans % mod
ans = (com(n*m -2, k-2) * ans) % mod

print(ans)