mod=10**9+7
def inv(x):
    if x==0:
        raise ValueError("Div 0")
    (nowx,lastx) = (0,1)
    (nowy,lasty) = (1,0)
    a=x;b=mod
    while b!=0:
        q=a//b
        (a,b)=(b,a%b)
        (nowx,lastx)=(lastx-q*nowx,nowx)
        (nowy,lasty)=(lasty-q*nowy,nowy)
    return (a*lastx)%mod
N,M,K=map(int,input().split())
X=sum([d*(N-d)*M*M for d in range(N)])
Y=sum([d*(M-d)*N*N for d in range(M)])
ans=X+Y
for i in range(1,K-1):
    ans*=(N*M-1-i)
    ans%=mod
    ans*=inv(i)
    ans%=mod
print(ans)
