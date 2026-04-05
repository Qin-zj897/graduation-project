def solve(nums):
    res=[]
    a=max(nums)
    b=min(nums)
    for num in nums:
        if num!=a and num!=b:
            res.append(num)
    return res


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
