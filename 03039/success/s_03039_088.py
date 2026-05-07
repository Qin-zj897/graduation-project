def comb(n, r):
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

N,M,K = map(int,input().split())

C = comb(N*M-2, K-2)
#print(int(C))

A = int(0)
for d in range(N):
    a = (N-d)*(M**2)

    A += int(d*a)

B = int(0)
for d in range(M):
    b = (M-d)*(N**2)
    B += int(d*b)

ans = int((A+B)*C)
ans = ans%(1000000007)
print(ans)
