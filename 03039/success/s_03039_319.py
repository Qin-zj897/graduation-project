import sys,heapq
from collections import deque,defaultdict
printn = lambda x: sys.stdout.write(x)
inn = lambda : int(input())
inl   = lambda: list(map(int, input().split()))
inm   = lambda:      map(int, input().split())
R = 10**9+7
DBG = True  and False
def ddprint(x):
  if DBG:
    print(x)

def exgcd(x,y):
    if y == 0:
        return (x,1,0)
    else:
        g,b,a = exgcd(y, x%y)
        return (g, a, b-(x//y)*a)

# return y s.t. x*y mod p == 1
def modinv(x,p):
    g,a,b = exgcd(x,p)  # ax+bp = g, g=1 if mutually prime
    return a%p

def comb(p,q):
    if q==0 or q==p:
        return 1
    if q > p-q:
        q = p-q
    num = p
    den = q
    for i in range(1,q):
        num = (num * (p-i))%R
        den = (den * (q-i))%R
    xden = modinv(den,R)
    return (num*xden)%R

n,m,k = inm()
x = ( (n*n*(m-1)*m*(m+1)+m*m*(n-1)*n*(n+1))//6 )%R
y = comb(n*m-2,k-2)
ddprint(x)
ddprint(y)
print((x*y)%R)
