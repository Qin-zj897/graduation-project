from collections import defaultdict,deque
import math
def gcd(a,b):
    return b if not a%b else gcd(b,a%b)
def lcm(a,b):
    return (a*b)//gcd(a,b)
def prepare(n, MOD):
    # 1! - n! の計算
    f = 1
    factorials = [1]  # 0!の分
    for m in range(1, n + 1):
        f *= m
        f %= MOD
        factorials.append(f)
    # n!^-1 の計算
    inv = pow(f, MOD - 2, MOD)
    # n!^-1 - 1!^-1 の計算
    invs = [1] * (n + 1)
    invs[n] = inv
    for m in range(n, 1, -1):
        inv *= m
        inv %= MOD
        invs[m - 1] = inv

    return factorials, invs
    #nCr = (factorials[n]%MOD * invs[r]%MOD * invs[n-r]%MOD)%MOD
def main():
    n,m,k=map(int, input().split())
    MOD = 10**9 + 7
    #k=2の場合nについて最大の差n-1は1通り、n-2は2通り・・・n-iはi通り(i<n)
    facs,invs = prepare(max(k,n*m),MOD)
    res = 0
    tmp = min(k,n)
    for i in range(1,n):
        res = res + ((n-i)*i*m*m)*(facs[n*m-2]%MOD * invs[k-2]%MOD * invs[n*m-2-(k-2)]%MOD)%MOD
        res%=MOD
    tmp = min(k,m)
    for i in range(1,m):
        res = res + ((m-i)*i*n*n)*(facs[n*m-2]%MOD * invs[k-2]%MOD * invs[n*m-2-(k-2)]%MOD)%MOD
        res%=MOD
    print(res)


if __name__ == '__main__':
    main()
