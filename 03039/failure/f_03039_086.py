N,M,K=map(int,input().split())
MOD=10**9+7

from fractions import Fraction

# 組合せの総数
def comb(n, k):
    
    if n < 0 or k < 0:
        pdt = 0
    
    else:
        pdt = 1
        for s in range(1, k + 1):
            pdt *= Fraction(n - s + 1, s)
    
    # 戻り値をint型にするために分子を取り出す
    return pdt.numerator


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