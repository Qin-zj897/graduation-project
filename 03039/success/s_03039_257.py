import sys
mod = pow(10, 9) + 7
def C(n, r):
    if ( r<0 or r>n ):
        return 0
    r = min(r, n-r)
    return g1[n] * g2[r] * g2[n-r] % mod
NNN = (10**6)
g1 = [1, 1]
g2 = [1, 1]
inverse = [0, 1]
for i in range(2, NNN+1):
    g1.append( ( g1[-1] * i ) % mod )
    inverse.append( ( -inverse[mod % i] * (mod//i) ) % mod )
    g2.append( (g2[-1] * inverse[-1]) % mod )
n,m,k=map(int,input().split())
#n=m=k=2
print(C(n*m,k)*C(k,2)%mod*(n+m)*inverse[3]%mod)
