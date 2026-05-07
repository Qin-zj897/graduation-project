n,m,k=map(int,input().split())
mod = 10**9+7
fac = [1]
caf = [1]
def pow(n,p):
    res=1
    while p >0:
        if p%2==0:
            n = n**2 % mod
            p //= 2
        else:
            res = res * n % mod
            p -= 1
    return res % mod

for i in range(1,n*m):
    fac.append((fac[i-1]*i)%mod)
    caf.append(pow(fac[i],mod-2))

def ncr(n,r):
    if n < r or r < 0: return 0
    return fac[n]*caf[r]*caf[n-r]

div2 = caf[2]

sums = [0]
for i in range((max(n,m))):
    sums.append((sums[-1]+i+1) % mod)

res1 = 0
for i in range(m):
    res1 += sums[i]+sums[m-1-i]
    res1 %= mod
res1 *= (n*n)%mod
res1 %= mod

res2 = 0
for i in range(n):
    res2 += sums[i]+sums[n-1-i]
    res2 %= mod
res2 *= (m*m)%mod
res2 %= mod

res = ((res1+res2)*div2)%mod
kc2 = ncr(n*m-2,k-2)
print((res*kc2)%mod)