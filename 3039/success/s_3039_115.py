def solve(nums):
    L=nums.copy()
    a=max(nums)
    b=min(nums)
    for i in nums:
        if i==a or i==b:
            L.remove(i)
    return L


if __name__ == '__main__':
    nums = (eval(input()))
    result = solve(nums)
    print(result)
