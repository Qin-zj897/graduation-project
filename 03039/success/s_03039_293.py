n,m,k=map(int,input().split())
mod=10**9+7
x=1
y=1
for i in range(1,k-1):
    y*=i
    y%=mod
for i in range(k-2):
    x*=n*m-2-i
    x%=mod
x*=pow(y,mod-2,mod)
x%=mod
def f(n,m):
    s=0
    for i in range(1,n):
        s+=i*(n-i)
        s%=mod
    return s*m*m%mod
print((f(n,m)+f(m,n))*x%mod)