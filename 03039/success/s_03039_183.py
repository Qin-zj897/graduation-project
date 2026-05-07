#cmb
mod=10**9+7
MAX=2*10**5+100

g1=[1,1]
g2=[1,1]
for i in range(2,MAX+1):
    num_1=g1[-1]*i%mod
    g1.append(num_1)
    g2.append(pow(num_1,mod-2,mod))
    
def cmb(n,r,MOD):
    return g1[n]*g2[r]*g2[n-r]%MOD

n,m,k=map(int,input().split())
s=0
t=0
for i in range(1,n):
    s+=i*(n-i)
for i in range(1,m):
    t+=i*(m-i)

ans=(n*n*t+m*m*s)%mod
ans=(ans*cmb(n*m-2,k-2,mod))%mod
print(ans)