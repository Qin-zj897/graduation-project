def cmb(n,r,mod):
  if r<0 or r>n:
    return 0
  r=min(r,n-r)
  return g1[n]*g2[r]*g2[n-r]%mod
 
n,m,k=map(int,input().split())
mod=10**9+7
 
g1=[1,1]
g2=[1,1]
inverse=[0,1]
 
for i in range(2,n*m):
  g1.append((g1[-1]*i)%mod)
  inverse.append((-inverse[mod%i]*(mod//i))%mod)
  g2.append((g2[-1]*inverse[-1])%mod)

c=cmb(n*m-2,k-2,mod)
ans=0
for i in range(n):
  ans+=((n-i)*m*m*i*c)%mod
  ans%=mod
for i in range(m):
  ans+=((m-i)*n*n*i*c)%mod
  ans%=mod
print(ans%mod)