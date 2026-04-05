def solve(nums):
    def search(nums):
        n=len(nums)
        for x in nums:
            if nums.count(x)>(n/2):
                y=x
            else:
                y=False
            return y





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
