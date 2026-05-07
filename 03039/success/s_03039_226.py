mod = 10**9+7
n,m,k = map(int,input().split())
##nCk:0,nにkが寄る時 O(k)

#nCkのkのmaxをxとする
x = k
fact = [1] * (x + 1)
for i in range(1, x + 1):
    fact[i] = fact[i - 1] * i % mod

def mod_comb_k(n,k,mod):
    ret = 1
    if n - k < k:
        k = n - k
    for i in range(n, n - k, -1):
        ret = ret * i % mod
    return ret % mod * pow(fact[k], mod - 2, mod) % mod

res = 0
for d in range(1,n):
    res += m*m*(n-d)*d
    res = res%mod

n,m = m,n
for d in range(1,n):
    res += m*m*(n-d)*d
    res = res%mod

res *= mod_comb_k(n*m-2,k-2,mod)
'''
for d in range(1,m):
    res += n*n*(m-d)*mod_comb_k(n*m-2,k-2,mod)%mod
    res = res%mod
'''
print(res%mod)