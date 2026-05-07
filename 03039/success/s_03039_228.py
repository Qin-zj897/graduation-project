from functools import reduce

def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

# mを法とするaの乗法的逆元
def modinv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        raise Exception('modular inverse does not exist')
    else:
        return x % m

MOD = 10**9+7

N,M,K = map(int,input().split())

patterns = reduce(lambda a,b: (a*b)%MOD, range(N*M-K+1, N*M+1))
patterns *= modinv(reduce(lambda a,b: (a*b)%MOD, range(1, K+1)), MOD)

patterns *= (N+M)*K*(K-1)
patterns *= modinv(6, MOD)
patterns %= MOD

print(patterns)