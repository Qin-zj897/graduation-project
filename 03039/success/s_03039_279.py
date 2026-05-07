def cmb(n, r, mod):
    if ( r<0 or r>n ):
        return 0
    r = min(r, n-r)
    return g1[n] * g2[r] * g2[n-r] % mod

A=2*10**5
mod = 10**9+7
g1 = [1, 1]
g2 = [1, 1]
inverse = [0, 1]

for i in range( 2, A + 1 ):
    g1.append( ( g1[-1] * i ) % mod )
    inverse.append( ( -inverse[mod % i] * (mod//i) ) % mod )
    g2.append( (g2[-1] * inverse[-1]) % mod )

N,M,K=map(int,input().split())
keisuu=cmb(N*M-2,K-2,mod)

sum_=0
for i in range(N):
    a=min(abs(i),abs(N-i-1))
    b=max(abs(i),abs(N-i-1))
    sum_+=(M**2)*((a*(a+1)//2)+(b*(b+1)//2))
for i in range(M):
    a=min(abs(i),abs(M-i-1))
    b=max(abs(i),abs(M-i-1))
    sum_+=(N**2)*((a*(a+1)//2)+(b*(b+1)//2))
print((keisuu * (sum_//2))%mod)