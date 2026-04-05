def solve(nums):
    def search(nums):
        n = len(nums)
        for i in nums:
            if nums.count(i)>(n/2):
                x = i
            else:
                x = False
        return(x)







    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
