def search(num):
    ns=[]
    for x in num:
        n=len(num)//2
        if num.count(x)>n and x not in ns:
            ns.append(x)
    if len(ns)>=1:
        return int(''.join(map(str,ns)))
    else: return False





nums = eval(input())
y = search(nums)
print(y)


