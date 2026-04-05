def solve(nums):
    def search(nums):
        lst1 = []
        icount = 0
        for x in nums:
            if nums.count(x)>len(nums)//2:
                lst1.append(x)
                break
        if len(lst1) == 0:
            return False
        else:
           return lst1[0]





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
