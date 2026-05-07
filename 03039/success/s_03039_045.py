mod=10**9+7
def mod_combination(n, k, mod):
    def ext_gcd(a, b):
        if b == 0:return a, 1, 0
        else:d,x,y = ext_gcd(b,a%b);x-=(a//b)*y 
        return d,y,x
    p,q=1,1
    for i in range(n-k+1,n+1):
        p=(p*i)%mod
    for i in range(2,k+1):
        q=(q*i)%mod
    return int(p*(ext_gcd(q, mod)[1]%mod)%mod)


n,m,k=map(int,input().split())
print(((pow(m,2,mod)*mod_combination(n*m-2,k-2,mod)*sum((n-i)*i%mod for i in range(1,n))%mod)+pow(n,2,mod)*mod_combination(n*m-2,k-2,mod)*sum((m-i)*i%mod for i in range(1,m))%mod)%mod)