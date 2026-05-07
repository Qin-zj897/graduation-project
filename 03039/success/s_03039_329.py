from collections import deque
from heapq import heappush,heappop
import re

def int_raw():
    return int(input())

def ss_raw():
    return input().split()

def ints_raw():
    return list(map(int, ss_raw()))

INF = 1<<29

DIV = 10**9+7

def mod_inv_prime(a,mod = DIV):
    return pow(a,mod-2,mod)

def mod_inv(a,b):
    r = a
    w = b
    u = 1
    v = 0
    while w!=0:
        t = r//w
        r -=t*w
        r,w = w,r
        u -= t*v
        u,v = v,u
    return (u%b+b)%b

def ncr(n,r, mod =DIV):
	r = min(r,n-r)
	ret = 1
	for i in range(1,r+1):
		ret = ret * (n-i+1) % mod
		ret = ret * mod_inv(i, mod) % mod
	return ret


def main():
    N, M,K = ints_raw()
    ans = 0
    for d in range(1,N):
        ans =(ans+d*(N-d)*M*M)%DIV
    for d in range(1,M):
        ans =(ans+d*(M-d)*N*N)%DIV
    return (ans*ncr(N*M-2,K-2))%DIV

if __name__ =="__main__":
    print(main())
