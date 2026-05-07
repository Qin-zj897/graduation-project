def cmb(n, r, mod):#コンビネーションの高速計算　
    if ( r<0 or r>n ):
        return 0
    r = min(r, n-r)
    return g1[n] * g2[r] * g2[n-r] % mod

mod = 10**9+7 #出力の制限
N = 2*10**5
g1 = [1, 1] # 元テーブル
g2 = [1, 1] #逆元テーブル
inverse = [0, 1] #逆元テーブル計算用テーブル

for i in range( 2, N + 1 ):
    g1.append( ( g1[-1] * i ) % mod )
    inverse.append( ( -inverse[mod % i] * (mod//i) ) % mod )
    g2.append( (g2[-1] * inverse[-1]) % mod )

def function(i,k):
    return -((k*(k+1)*(2*k+1))//3)+k*(k+1)*(N+i-M)+2*N*(k+1)*(M-i)

N,M,K=map(int,input().split())
if M>N:
    N,M=M,N
ans=0
for i in range(1,N+M-1):
    if M-1>=i:
        ans+=i*(function(i,i)-N*(M-i)-M*(N-i))
        ans=ans%mod
    if i>=M and N-1>=i:
        ans+=i*(function(i,i)-function(i,i-M)-M*(N-i))
        ans=ans%mod
    if i>=N:
        ans+=i*(function(i,N-1)-function(i,i-M))
        ans=ans%mod

ans=ans*cmb(N*M-2,K-2,mod)
ans=ans%mod
print(ans)