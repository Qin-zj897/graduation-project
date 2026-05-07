n,m,k=map(int,input().split())

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

mod=10**9+7
a=cmb(n*m-2,k-2)%mod
res=0
for d in range(1,n):
	res+=(d*(n-d)*m**2*a)%mod
	res%=mod
for d in range(1,m):
	res+=(d*(m-d)*n**2*a)%mod
	res%=mod
print(res)