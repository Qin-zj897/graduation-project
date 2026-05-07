def modinv(a):
    b=MOD
    (x, lastx) = (0, 1)
    (y, lasty) = (1, 0)
    while b != 0:
        q = a // b
        (a, b) = (b, a % b)
        (x, lastx) = (lastx - q * x, x)
        (y, lasty) = (lasty - q * y, y)
    return lastx%MOD

MOD=10**9+7
N,M,K=map(int,input().split())

b=1
for i in range(1,K-1):
    b=(b*(N*M-1-i)*modinv(i))%MOD

x=0
for i in range(1,N):
   x=x+i*(N-i)
x=(x*M*M)%MOD

y=0
for j in range(1,M):
    y=y+j*(M-j)
y=(y*N*N)%MOD
    
ans=b*(x+y)%MOD
print(ans)