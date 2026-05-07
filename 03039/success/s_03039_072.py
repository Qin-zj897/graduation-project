N,m,k = (int(i) for i in input().split())
n,mod,x,y = N*m,10**9+7,0,0
fn,fk = [1]*n,[1]*n
for i in range(n-1): fn[i+1] = (fn[i]*(i+2))%mod
def power(n,k):
	if k==1: return n
	elif k%2==0: return power((n**2)%mod,k//2)
	else: return (n*power(n,k-1))%mod
def comb(n,k):
	if n==0 or k==0 or n==k: return 1
	else: return (((fn[n-1]*fk[n-k-1])%mod)*fk[k-1])%mod
fk[-1] = power(fn[-1],mod-2)
for i in range(2,n+1): fk[-i] = (fk[-i+1]*(n+2-i))%mod
for i in range(N): x+=i*(N-i)*(m**2)
for i in range(m): y+=i*(m-i)*(N**2)
print(comb(n-2,k-2)*(x+y)%mod)