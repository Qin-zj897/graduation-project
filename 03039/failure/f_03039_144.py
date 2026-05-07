def getInv(N):#Qはmod
    inv = [0] * (N + 1)
    inv[0] = 1
    inv[1] = 1
    for i in range(2, N + 1):
        inv[i] = (-(Q // i) * inv[Q%i]) % Q
    return inv

Q = 10**9+7
modinv = getInv(4*10**5)
N, M, K = map( int, input().split())
T = N*M
comb = 1
for i in range(K-2):
    comb *= T-2-i
    comb %= Q
    comb *= modinv[K-2-i]
    comb %= Q
ans = (N**2)%Q*(N-1)*(N+1)%Q*M%Q*modinv[6]%Q + (M**2)%Q*(M-1)*(M+1)%Q*N%Q*modinv[6]%Q
ans %= Q
# print(ans,T-2, K-2, comb)
print( ans*comb%Q)