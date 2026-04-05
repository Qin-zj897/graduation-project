def solve(nums):
    def search(nums):
        for x in nums:
            a=nums.count(x)
            n=len(nums)
            if a>n//2:
                return x
            if a<=n//2:
                return False






    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
