def mod_exp(b,e,mod):
    r = 1
    while e > 0:
        if (e&1) == 1:
            r = (r*b)%mod
        b = (b*b)%mod
        e >>= 1

    return r

# get degree of p in n! (exponent of p in the factorization of n!)
def fact_exp(n,p):
    e = 0
    u = p
    t = n
    while u <= t:
        e += t//u
        u *= p
    return e

# convert given number n into array of its base b representation
# most significant digit is at rightmost position in array
def get_base_digits(n,b):
    d = []
    while n > 0:
        d.append(n % b)
        n  = n // b
    return d

# Using Fermat's little theorem to compute nCk mod p
# considering cancelation of p in numerator and denominator
# Note: p must be prime
def fermat_binom_advanced(n,k,p):
    # check if degrees work out
    num_degree = fact_exp(n,p) - fact_exp(n-k,p)
    den_degree = fact_exp(k,p)
    if num_degree > den_degree:
        return 0
    if k > n:
        return 0
    # calculate numerator and cancel out occurrences of p
    num = 1
    for i in range(n,n-k,-1):
        cur = i
        while cur%p == 0:
            cur //= p
        num = (num*cur)%p
    # calculate denominator and cancel out occurrences of p
    denom = 1
    for i in range(1,k+1):
        cur = i
        while cur%p == 0:
            cur //= p
        denom = (denom*cur)%p
    # numerator * denominator^(p-2) (mod p)
    return (num * mod_exp(denom,p-2,p))%p

# Using Lucas' theorem to split the problem into smaller sub-problems
# In this version assuming p is prime
def lucas_binom(n,k,p):
    # get n and k in base p representation
    np = get_base_digits(n,p)
    kp = get_base_digits(k,p)
    # calculate (nCk) = (n0 choose k0)*(n1 choose k1) ... (ni choose ki) (mod p)
    binom = 1
    for i in range(len(np)-1,-1,-1):
        ni = np[i]
        ki = 0
        if i < len(kp):
            ki = kp[i]
        binom = (binom * fermat_binom_advanced(ni,ki,p)) % p
    return binom

mod = 10**9+7

n,m,k = map(int,input().split())
temp = 0
for d in range(1,m):
	temp += d*n*n*(m-d)
for d in range(1,n):
	temp += d*m*m*(n-d)
l = lucas_binom(n*m-2,k-2,mod)
print(temp*l%mod)