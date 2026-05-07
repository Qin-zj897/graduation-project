facs = []
M = 1000000007

def modulo(n):
    global facs
    facs = [1]*(n+1)
    last = 1
    for i in range(1,n+1):
        last = facs[i] = mulmod(last,i,M)
        
def mulmod(x,y,p):
    return x*y % p
def divmod(x,y,p):
    return mulmod(x,powmod(y,p-2,p),p)

def ncr(n,r):
    if n<r:
        return 0
    if n==r:
        return 1
    res = facs[n]
    res = divmod(res,facs[r],M)
    res = divmod(res,facs[n-r],M)
    return res

def powmod(x,y,p):
    if y==0:
        return 1
    elif y==1:
        return x % p
    elif (y%2)==0:
        return powmod(x,y//2,p)**2 % p
    else:
        return powmod(x,y//2,p)**2 * x % p
def resolve():
    n,m,k = map(int,input().split())
    modulo(2*10**5+5)
    kumi=ncr(n*m-2,k-2)
    ans = 0
    for i in range(1,n):
        ans = ans + ((i*(n-i)*m*m)%M)
    for i in range(1,m):
        ans = ans + ((i*(m-i)*n*n)%M)
    print(ans*kumi%M)

if __name__ == "__main__":
    resolve()