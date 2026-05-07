def gcd(a, b):
    if b == 0:
        return a
    else:
        return gcd(b, a % b)

def opt(a, b):
    c = gcd(a, b)
    return a//c, b//c

def dot(A, B):
    a = A[0] * B[0]
    b = A[1] * B[1]
    return opt(a, b)

def add(A, B):
    return opt(A[0]*B[1] + B[0]*A[1], A[1] * B[1])

N, A, B, C = map(int, input().split())

A = opt(A, 100)
B = opt(B, 100)
C = opt(C, 100)

g = 1000000007

c = [((1, 1), 0, 0, 0)]
ans = (0, 1)
e = (0, 1)
pi = 0
while pi < len(c):

    PQ, cnt, acnt, bcnt = c[pi]
    pi += 1

    if e[1] > (e[1] - e[0]) * g * 10:
        continue
    if acnt >= N or bcnt >= N:
        e = add(e, PQ)
        ans = add(ans, opt(cnt * PQ[0], PQ[1]))
        continue

    cnt += 1
    c.append((dot(PQ, A), cnt, acnt+1, bcnt))
    c.append((dot(PQ, B), cnt, acnt, bcnt+1))
    c.append((dot(PQ, C), cnt, acnt, bcnt))


print(ans[0]/ans[1])

