Mod = 10**9 + 7

n,m,k = map(int,input().split())

inv = [0]*(k+3)
inv[1] = 1
for i in range(2,k+2):
  # print(i)
  inv[i] = ((Mod-Mod//i) * inv[Mod%i]) % Mod


s = (n*n*(m+1)*m*(m-1) + m*m*(n+1)*n*(n-1))//6
for i in range(k-2):
   s = (s*(n*m-2-i)*inv[i+1])%Mod
print(s)