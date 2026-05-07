n=int(input())
p=[]
total=0
for i in range(n,0,-1):
    p.append(i)
p.reverse()
for i in range(1,n):
    total+=i%p[i]
print(total)

