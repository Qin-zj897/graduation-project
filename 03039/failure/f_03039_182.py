N,M,K=map(int,input().split())
O=10**9+7
S=N+M
for i in range(K):S=S*(N*M-i)*pow(i+1,O-2,O)
print(S*K*~-K*-~O//6%O)