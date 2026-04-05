def solve(nums):
    a = max(nums)
    b = min(nums)
    kong = []
    for i in nums:
        if i == a or i == b:
            pass
        else:
            kong.append(i)
    return kong


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
