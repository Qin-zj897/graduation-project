N,M,K=map(int,input().split())
O=10**9+7
f=lambda n:1if n<1else n*f(n-1)%O;
print(f(N*M-2)*pow(6*f(N*M-K)*f(K-2),O-2,O)*N*M*(N*M-1)*(N+M)%O)