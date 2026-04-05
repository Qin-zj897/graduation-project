def solve(nums):
    def search(nums):
        n=len(nums)
        for i in nums:
            if nums.counts(i)>n//2:
                i=y
            else:
                return False
             return y





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
