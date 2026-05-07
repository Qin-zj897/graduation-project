def pmod(a,b,m):
  ret=1
  while b>0:
    if b&1:
      ret=(ret*a)%m
    a=(a*a)%m
    b=b>>1
  return ret
  
def fmod(n,m):
  ret=1
  for i in range(n,0,-1):
    ret=(ret*i)%m
  return ret

def cmod(n,r,m):
  ret=1
  for i in range(r):
    ret=(ret*(n-i))%m
  return (ret*pmod(fmod(r,m),m-2,m))%m

mod=10**9+7
n,m,k=map(int,input().split())
ans=0
c=cmod(n*m-2,k-2,mod)
for i in range(1,n):
  ans+=i*c*(m**2)*(n-i)
  ans=ans%mod
for i in range(1,m):
  ans+=i*c*(n**2)*(m-i)
  ans=ans%mod
print(ans%mod)