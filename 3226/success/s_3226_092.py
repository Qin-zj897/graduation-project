def solve(nums):
    def search(nums):
        for i in nums:
            if nums.count(i)>(len(nums)/2):
                h=i
            else:
                h=False
        return h





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
