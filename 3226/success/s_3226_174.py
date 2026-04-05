def solve(nums):
    def search(nums):
        for i in nums:
            c=nums.count(i)
            if c>0.5*len(nums):
                return i
            else:
                return "False"





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
