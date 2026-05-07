from functools import reduce
p = 1000000007
def powmod(a, n):
    global p
    n_b = str(format(n, "b"))  # 2進表現に
    res = 1
    for bit in n_b:
        res = (res * res) % p
        if bit == "1":
            res = (res * a) % p
    return res
def cmod(n, r):
    if r > n // 2:
        return cmod(n, n - r)
    def mulmod(x, y):
        global p
        return x * y % p
    return powmod(reduce(mulmod, range(n - r + 1, n + 1), 1) * reduce(mulmod, range(1, r + 1), 1), (p - 2))
n, m, k = map(int, input().split())
print(((n**3-n)*m**2+(m**3-m)*n**2)//6*cmod(n+m-2,k-2)%p)