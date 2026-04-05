def solve(nums):
    def search(nums):
        for x in nums:
            c=nums.count(x)
            if c>0.5*len(nums):
                return x
            else:
                return "False"






    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
