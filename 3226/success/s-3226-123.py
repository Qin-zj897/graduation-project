def search(nums):
    i=0
    nums1=list(set(nums))
    for x in nums1:
        if nums.count(x)> len(nums)//2:
            return x
            i=i+1
    if i==0:
        return False





nums = eval(input())
y = search(nums)
print(y)


