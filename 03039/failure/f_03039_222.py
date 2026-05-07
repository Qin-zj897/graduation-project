def com(n,r):
    if n-r<r:
        r=n-r
    re=1
    for i in range(n,n-r,-1):
        re*=i
    for i in range(1,r+1):
        re//=i
    return re

def f(n,m,k):
    md = 10 ** 9 + 7
    ans = 0
    for d in range(1, m):
        ans = (ans + d * (m - d) * n * n) % md
    for d in range(1, n):
        ans = (ans + d * (n - d) * m * m) % md
    ans = (ans * com(n * m - 2, k - 2)) % md
    print(ans)

n,m,k=map(int,input().split())
f(n, m, k)
