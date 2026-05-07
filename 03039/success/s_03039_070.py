
#modの掛け算

def modmal(a,b,mod): #a*bをmodを法にして求める

    return a * b % mod


#modの割り算

def moddiv(a,b,mod): #a/bをmodを法にして求める

    return (a * pow(b,mod-2,mod)) % mod


#逆元

def inverse(a,mod): #aのmodを法にした逆元を返す
    return pow(a,mod-2,mod)



#modのn!と、n!の逆元を格納したリストを返す(拾いもの)
#factorialsには[1, 1!%mod , 2!%mod , 6!%mod… , n!%mod] が入っている
#invsには↑の逆元が入っている

def modfac(n, MOD):
 
    f = 1
    factorials = [1]
    for m in range(1, n + 1):
        f *= m
        f %= MOD
        factorials.append(f)
    inv = pow(f, MOD - 2, MOD)
    invs = [1] * (n + 1)
    invs[n] = inv
    for m in range(n, 1, -1):
        inv *= m
        inv %= MOD
        invs[m - 1] = inv
    return factorials, invs


def modnCr(n,r,mod,fac,inv): #上で求めたfacとinvsを引数に入れるべし(上の関数で与えたnが計算できる最大のnになる)

    return fac[n] * inv[n-r] * inv[r] % mod

"""
とある二つのxのみに着目すると
距離0になるのは N * M * (M-1)通り
距離xになるのは (N-x) * M * M 通り
すべての場合において他の駒のおき方が (N*M-2)! / (N*M-K)! 通りある
駒を区別しないので駒の割り当て方→K!でわる

110 1
101 2
011 1
"""

N,M,K = map(int,input().split())
mod = 10**9+7

fac,inv = modfac(N*M+1,mod)


ansx = 0

for dis in range(N):

    if dis == 0:
        ansx += N*M*(M-1) * dis
    else:
        ansx += 2 * (N-dis) * M * M * dis

    ansx %= mod

ansx *= fac[N*M-2] * inv[N*M-K]
ansx %= mod
ansx *= modnCr(K,2,mod,fac,inv)
ansx *= inv[K]

ansy = 0
for dis in range(M):

    if dis == 0:
        ansy += M*N*(N-1) * dis
    else:
        ansy += 2 * (M-dis) * N * N * dis

    ansy %= mod

ansy *= fac[N*M-2] * inv[N*M-K]
ansy %= mod
ansy *= modnCr(K,2,mod,fac,inv)
ansy *= inv[K]

print ((ansx+ansy)%mod)    