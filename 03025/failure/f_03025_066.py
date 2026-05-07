N,A,B,C = map(int, input().split())

mod = 10**9+7
n = 100*N
d = A+B


def modinv(a,p):
    b = p
    u = 1
    v = 0
    while b:
        t = a//b
        a -= t*b
        a,b = b,a
        u -= t*v
        t,v=v,u
    u %= p
    if u < 0:
        u += p
    return u

n %= mod
print(n*modinv(d,mod)%mod)