def solve(nums):
    def search(nums):
        le=len(nums)//2
        for x in nums:
            if nums.count(x)>le:
                return x
            else:
                return






    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
