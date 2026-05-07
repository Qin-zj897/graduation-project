def cmb(n, r):
    if n - r < r: r = n - r
    if r == 0: return 1
    if r == 1: return n

    numerator = [n - r + k + 1 for k in range(r)]
    denominator = [k + 1 for k in range(r)]

    for p in range(2,r+1):
        pivot = denominator[p - 1]
        if pivot > 1:
            offset = (n - r) % p
            for k in range(p-1,r,p):
                numerator[k - offset] /= pivot
                denominator[k] /= pivot

    result = 1
    for k in range(r):
        if numerator[k] > 1:
            result *= int(numerator[k])
    return result

def gcd(a, b):
    while b:
        a, b = b, a%b
    return a


def reduce(p, q):
    common = gcd(p, q)
    return (p // common, q // common)

n,a,b,c = map(int,input().split())
d = a + b
a1 = a ** n
b1 = b ** n
e = 0
for i in range(n,2*n):
    m1 = i - n
    m2 = n - m1
    o = cmb(i-1,m1) 
    a2 = a ** m1
    b2 = a ** m1
    x1 = a1 * b2 * (d ** m2) * o
    x2 = a2 * b1 * (d ** m2) * o
    e += x1 + x1

p = e * 100
q = (d ** (2*n)) * c
u = reduce(p,q)
p = u[0]
q = u[1] % (10**9 +7)
print(p // q)