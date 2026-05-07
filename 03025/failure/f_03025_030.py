n,a,b,c=(int(i) for i in input().split())
a=a/100
b=b/100
def bitsu(n):
  r=1
  for i in range(n):
    r=r*(i+1)
  return r
def comb(n,k):
  return bitsu(n)/(bitsu(k)*bitsu(n-k))
def pos(a,b,k): #k回目で終わる確率
  return (a**n)*(b**(k-n))*comb(n,k)+(a**(k-n))*(b**n)*comb(n,k)
r=0
for i in(n,2*n-1):
  r=r+i*pos(a,b,i)
print(r)