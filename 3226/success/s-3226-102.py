def search(nums):
    dic={}
    n=len(nums)//2
    for x in nums:
        dic[x]=dic.get(x,0)+1
    maxv=max(dic.values())
    if maxv>n:
        for x in dic.keys():
            if dic[x]==maxv:
                return x
    else:
        return 'False'





nums = eval(input())
y = search(nums)
print(y)


