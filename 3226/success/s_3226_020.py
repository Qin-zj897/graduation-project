def solve(nums):
    def search(nums):
        n = len(nums)
        for x in nums:
            if nums.count(x)>(n//2):
                m = x
            else:
                m = False
        return m





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
