def solve(nums):
    def search(nums):
        for x in nums:
            if nums.count(x)>len(nums)/2:
                h=x
                return h
            else:
                h=False
        return h    





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
