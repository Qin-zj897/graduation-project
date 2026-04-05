def solve(nums):
    max = max(nums)
    min = min(nums)
    x = nums.copy()
    for n in nums:
        if n == max or n == min:
            x.remove(n)
    return x


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
