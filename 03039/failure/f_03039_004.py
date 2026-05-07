n,m,k = map(int, input().split())
mod = 10**9+7
def cc(a,b):
  b = min(b,a-b)
  tot = 1
  for i in range(1,b+1):
    tot = (tot*(a-i+1)//i)%mod
  return tot
xans = 0
yans = 0
for i in range(1,n):
  xans+=i*(n-i)*m**2
  xans%=mod
for j in range(1,m):
  yans+=i*(m-i)*n**2
  yans%=mod
print((xans+yans)*cc(n*m-2,k-2)%mod)