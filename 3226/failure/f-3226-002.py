def search(nums):
    dic={}
    for i in nums:
        if i not in search:
            search[i]=1
        else:
            search[i]+=1
    return max(search.values())






nums = eval(input())
y = search(nums)
print(y)


