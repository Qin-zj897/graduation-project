MOD=10**9+7
UPPERLIMIT=3*10**5
MODMUL=[1, 1]+[0]*(UPPERLIMIT-1)
for i in range(2, UPPERLIMIT+1):
  MODMUL[i]=MODMUL[i-1]*i%MOD
MODDIV=[1]*UPPERLIMIT+[pow(MODMUL[-1], MOD-2, MOD)]
for i in range(UPPERLIMIT, 0, -1):
  MODDIV[i-1]=MODDIV[i]*i%MOD
def MODCOMB(n, r):
  if n<r or n<1:
    return 0
  else:
    return (((MODMUL[n]*MODDIV[r])%MOD)*MODDIV[n-r])%MOD

N, M, K=map(int, input().split())

ans=0
k=MODCOMB(N*M-2, K-2)
P=min(N, M)
Q=max(N, M)

for sa in range(1, P):
  ans=(ans+sa*(P-sa)*M*M*k)%MOD
  ans=(ans+sa*(P-sa)*N*N*k)%MOD
  
for sa in range(P, Q):
  ans=(ans+sa*(Q-sa)*P*P*k)%MOD

print(ans)