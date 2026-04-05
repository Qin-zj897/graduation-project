def solve(nums):
    def search(nums:list):
        r=False
        l=len(nums)
        for x in nums:
            c=nums.count(x)
            if c>l/2:
                r=x
        return r





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
