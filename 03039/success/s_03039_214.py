N, M, K = map(int, input().split())
ppp = 10**9 + 7

#def expo_div_p(a: int,n: int,p: int):
#    if n == 1:
#        d = a
#    elif n % 2 == 0:
#        d = expo_div_p(a, n//2, p)**2 % p
#    else:
#        d = a * expo_div_p(a, n//2, p)**2 % p
#    return d
#
def combi(x, y, p):
    c = 1
    for i in range(1,y+1):
        c *= (x-i+1) * pow(i, p-2, p)
        c %= p
    return c


ans = 0
C = combi(N*M-2, K-2, ppp)

for i in range(1,N):
    ans += C * i * (N-i) * M**2
for j in range(1,M):
    ans += C * j * (M-j) * N**2

#print(C)
print(ans % ppp)
