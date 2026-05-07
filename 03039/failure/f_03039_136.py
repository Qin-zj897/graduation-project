def reduce_mul(s,e):
     a = 1
     for i in range(s,e+1):
          a = a*i
     return a
     
def cmb(n,r):
    r = min(n-r,r)
    if r == 0: return 1
    over = reduce_mul(n-r+1,n)
    under = reduce_mul(1,r)
    return over // under

n,m,k = map(int,input().split())
nm = n*m%(10**9+7)
k = k%(10**9+7)
a = (cmb(nm-2,k-2)%(10**9+7))*(nm*(m*(n**2-1)+n*(m**2-1)))//6
print(a%(10**9+7))