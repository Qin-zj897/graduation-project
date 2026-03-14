def search(nums):
    for i in range(len(nums)):
        if nums.count(i)>((len(nums))/2):
            return i
        else:
            continue
    return False





nums = eval(input())
y = search(nums)
print(y)


