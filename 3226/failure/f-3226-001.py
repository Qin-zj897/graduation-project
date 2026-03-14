ls = eval(input())
n = eval(input())
ls2 = ls*n
ls3 = [x*x for x in ls2]
ls4=[]
for x in ls3:
    if x not in ls4:
        ls4.append(x)
print(ls4)





nums = eval(input())
y = search(nums)
print(y)


