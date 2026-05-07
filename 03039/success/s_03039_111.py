from collections import defaultdict,deque
import sys,heapq,bisect,math,itertools,string,queue,datetime
sys.setrecursionlimit(10**8)
INF = float('inf')
mod = 10**9+7
eps = 10**-7
def inpl(): return list(map(int, input().split()))
def inpl_str(): return list(input().split())


H,W,K = inpl()

MAX = H*W+5
fac = [1]*(MAX+1)
for i in range(1,MAX+1):
	fac[i] = (fac[i-1]*i)%mod

gyakugen = [1]*(MAX+1)
gyakugen[MAX] = pow(fac[MAX],mod-2,mod)
for i in range(MAX,0,-1):
	gyakugen[i-1] = (gyakugen[i]*i)%mod

def Comb(n,k):#nCk
	return (fac[n]*gyakugen[k]*gyakugen[n-k])%mod

ans = 0
ruiseki = [0]
tmp = 0
for i in range(1,max(H,W)+5):
    tmp += i
    ruiseki.append(tmp)

for y in range(H):
    ans += (ruiseki[y] + ruiseki[H-y-1])*W*W

for x in range(W):
    ans += (ruiseki[x] + ruiseki[W-x-1])*H*H

ans //= 2
ans *= Comb(H*W-2,K-2)

print(ans%mod)
