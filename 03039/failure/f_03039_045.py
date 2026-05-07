p = 10**9+7
N,M,K = map(int,input().split())
a = ((K*(K-1))//2)%p
b = (M**2)%p
b = (b*(((N*(N-1))//2)%p))%p
c = (N**2)%p
c = (c*((M*(M-1))//2)%p)%p
print((a*(b+c))%p)