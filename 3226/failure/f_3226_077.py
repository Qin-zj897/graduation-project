def solve(nums):
    def search(nums):
        s=len(nums)//2
        for i in nums:
            if nums.counts(i)>s:
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
