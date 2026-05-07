n,m,k=map(int,input().split())
mod=pow(10,9)+7
ans=0

#n*m-2個の中からk-2個選ぶときの組み合わせ数
cmb=1
for i in range(k-2):
  cmb*=n*m-i-2
  cmb%=mod
  cmb*=pow(i+1,mod-2,mod)
  cmb%=mod
for i in range(1,n):
  tmp=i*(n-i)
  tmp*=m**2
  tmp%=mod
  tmp*=cmb
  tmp%=mod
  ans+=tmp
  ans%=mod
for i in range(1,m):
  tmp=i*(m-i)
  tmp*=n**2
  tmp%=mod
  tmp*=cmb
  tmp%=mod
  ans+=tmp
  ans%=mod
print(ans)