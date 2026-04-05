def solve(nums):
    max_num=max(nums)
    min_num=min(nums)
    tmp=nums.copy()
    for num in nums:
        if num==max_num or num==min_num:
            tmp.remove(num)
    return tmp


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
