import math

N, M, K = (int(x) for x in input().split())

def comb(n,k,p):
    if k < 0 or k > n:
        return 0
    if n == 0 or k == 0:
        return 1
    a=math.factorial(n) % p
    b=math.factorial(k) % p
    c=math.factorial(n-k) % p
    return (a*mypow(b,p-2,p)*mypow(c,p-2,p)) % p

def mypow(a,b,p):
    if b == 0: return 1
    if b % 2 == 0:
        d=mypow(a,b//2,p)
        return d*d % p
    if b % 2 == 1:
        return (a*mypow(a,b-1,p )) % p

c_sum = (M*(N-1)*N*(N+1)*M+N*(M-1)*M*(M+1)*N)//3
c_sum = c_sum//2
c_sum *= comb(N*M-2, K-2, 1000000007)
c_sum = c_sum % 1000000007

print(c_sum)
