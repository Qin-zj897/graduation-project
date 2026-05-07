def cmb(n,r,mod):
  if r<0 or r>n:
    return 0
  r=min(r,n-r)
  return g1[n]*g2[r]*g2[n-r]%mod
 
N,M,k=map(int,input().split())
m=10**9+7
 
g1=[1,1]
g2=[1,1]
inverse=[0,1]
 
for i in range(2,N*M):
  g1.append((g1[-1]*i)%m)
  inverse.append((-inverse[m%i]*(m//i))%m)
  g2.append((g2[-1]*inverse[-1])%m)

c=cmb(N*M-2,k-2,m)
ans=0
for i in range(N):
  ans+=((N-i)*M*M*i*c)%m
  ans%=m
for i in range(M):
  ans+=((M-i)*N*N*i*c)%m
  ans%=m
print(ans%m)