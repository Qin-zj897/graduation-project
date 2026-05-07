
"""
xi,xjのペアにおいて、このペアは何通り出てくるか
sum(全通り)sum(i)sum(j){|xi-xj|+|yi-yj|}
和なので、どんな順序でもいい

sum(i)sum(j)sum(全通り){|xi-xj|}
sum(i)sum(j)sum(全通り){|yi-yj|}

あるi,jで固定して、sum(全通り){|xi-xj|}を求めるヤツ
固定したi,jが現れる回数は、

pat = NM-2_C_K-2 こいつはどこでも一緒

pat*sum(i)sum(j){|xi-xj|}を計算する
すべてのi,j(i>j)ペアにおいて、xi,xjの和は
sum(i=1,N){i*(N-i)}
"""
from math import factorial

N,M,K = map(int,input().split())
mod = 10**9+7
fa = 1
num = N*M - 2
# 固定したときの通り数
# フェルマーの小定理をつかって、mod(M)でのcombinationが高速で求まる
# http://keita-matsushita.hatenablog.com/entry/2016/12/05/184011
def mod_combination(n,r,mod):
    if n<0 or r<0 or n<r:
        return 0
    if n==0 or r == 0:
        return 1
    a = factorial(n) % mod
    b = factorial(r) % mod
    c = factorial(n-r) % mod
    return (a*pow(b,mod-2,mod)*pow(c,mod-2,mod)) % mod 

pat = mod_combination(N*M-2,K-2,mod)

n_x = M**2*((N-1)*N*(N+1))//6
n_y = N**2*((M-1)*M*(M+1))//6
print(pat*(n_x+n_y)%mod)