def search(nums):
    dic={}
    for i in nums:
        if i not in dic:
            dic[i]=1
        else:
            dic[i]+=1
    return max(dic.values())





nums = eval(input())
y = search(nums)
print(y)


