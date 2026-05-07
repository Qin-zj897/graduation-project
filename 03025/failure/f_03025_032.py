n,a,b,c=(int(i)for i in input().split())
if a>b:
  d=a
else:
  d=b
e=n/(d/100)
for i in range(10**9):
  if ((i+1)*d)%(10**9+7)==(100*n)%(10**9+7):
    print(i+1)
    break