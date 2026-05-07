N,M,K=map(int,input().split())
MOD=10**9+7

# 組合せの総数
def comb(n, r):
    if n - r < r: r = n - r
    if r == 0: return 1
    if r == 1: return n

    numerator = [n - r + k + 1 for k in range(r)]
    denominator = [k + 1 for k in range(r)]

    for p in range(2,r+1):
        pivot = denominator[p - 1]
        if pivot > 1:
            offset = (n - r) % p
            for k in range(p-1,r,p):
                numerator[k - offset] /= pivot
                denominator[k] /= pivot

    result = 1
    for k in range(r):
        if numerator[k] > 1:
            result *= int(numerator[k])
            result%=MOD

    return result



ans_x=0
ans_y=0

for i in range(1,N):
  ans_x+=i*(N-i)
  ans_x%=MOD
for i in range(1,M):
  ans_y+=i*(M-i)
  ans_y%=MOD
ans_x*=((M**2))
ans_y*=((N**2))
ans=comb(N*M-2,K-2)
ans*=(ans_x+ans_y)
ans%=MOD
print(int(ans))