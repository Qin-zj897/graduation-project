def comb(n, k):
    x, y = 1, 1
    k = min(k, n-k)
    for i in range(k):
        x *= n - i
        x %= MOD
        y *= i + 1
        y %= MOD
        
    return x*modinv(y) % MOD

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


x=0
for i in range(1,N//2+1):
   x+=2*i*(N-i)
if N%2==0: x-=N*N//4

y=0
for j in range(1,M//2+1):
    y+=2*j*(M-j)
if M%2==0: y-=M*M//4 

ans=comb(N*M-2,K-2)*(x*M*M+y*N*N)%MOD
print(ans)