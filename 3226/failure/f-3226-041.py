ls = []
def search(nums):
    n = len(nums)
    for x in nums:
        if nums.count(x) > n/2:
            ls.append(x)
            print(x)
        else:
            pass
    if ls == []:
        print("False")
    else:
        pass





nums = eval(input())
y = search(nums)
print(y)


