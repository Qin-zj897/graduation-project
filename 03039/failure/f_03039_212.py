n,m,k=map(int,input().split())
sn=0
sm=0
for i in range(n+1):
  sn+=(i*(i+1)+(n-i)*(n-i+1))//2
for i in range(m+1):
  sm+=(i*(i+1)+(m-i)*(m-i+1))//2
sn=sn*((m+1)**2)
sm=sm*((n+1)**2)
c=k*(k-1)/2
sn=(sn*c)%(10**9+7)
sm=(sm*c)%(10**9+7)
print((sn+sm)%(10**9+7))