def egcd(a, b):
    (x, lastx) = (0, 1)
    (y, lasty) = (1, 0)
    while b != 0:
        q = a // b
        (a, b) = (b, a % b)
        (x, lastx) = (lastx - q * x, x)
        (y, lasty) = (lasty - q * y, y)
    return (lastx, lasty, a)

# ax ≡ 1 (mod m)なるxを返す
def modinv(a):
    m=MOD
    (inv, q, gcd_val) = egcd(a, m)
    return inv % m

MOD=10**9+7
N,M,K=map(int,input().split())

b=1
for i in range(N*M-2,N*M-K,-1):
    b=(b*i)%MOD

for i in range(1,K-1):
    b=(b*modinv(i))%MOD

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