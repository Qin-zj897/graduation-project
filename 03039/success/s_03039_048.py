import sys
import bisect
from collections import defaultdict

ans=0
#n=int(input())
n,m,k=map(int,input().split())
#a=[(1,int(i)) for i in input().split()]

# http://drken1215.hatenablog.com/entry/2018/06/08/210000
max_value = n*m+2
mod = 10 ** 9 + 7

fac = [0 for _ in range(max_value)]
finv = [0 for _ in range(max_value)]
inv = [0 for _ in range(max_value)]

def com_init():
    fac[0] = 1
    fac[1] = 1
    finv[0] = 1
    finv[1] = 1
    inv[1] = 1

    for i in range(2, max_value):
        fac[i] = fac[i - 1] * i % mod
        inv[i] = mod - inv[mod % i] * (mod // i) % mod
        finv[i] = finv[i - 1] * inv[i] % mod

def com(n, k):
    if n < k:
        return 0
    if n < 0 or k < 0:
        return 0
    return fac[n] * (finv[k] * finv[n - k] % mod) % mod

def dist(a,b):
    if a<=0 or b<=0:
        return 0
    
    a,b=min(a,b),max(a,b)
    a,b=a%mod,b%mod
    '''
    rt=((a*(a+1))%mod)*(a+b)//2
    rt+=((a*(a+b))%mod)*(b-a-1) if a!=b else 0
    rt+= - a*a if a==b else 0
    '''
    rt=((b*a*(a+b))//2)%mod
    #rt//=2
    return rt%mod

com_init()
inv2=500000004
#print(ans,dist(2,2),dist(1,0))
d=0
for i in range(n):
    for j in range(m):
        d+=dist(j,i+1)+dist(i,m-j)+dist(j+1,n-i-1)+dist(n-i,m-j-1)
        d%=mod
ans=d*com(n*m-2,k-2)
ans*=inv2

print(ans%mod)
        
        

