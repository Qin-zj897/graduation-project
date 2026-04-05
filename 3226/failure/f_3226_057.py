def solve(nums):
    def search(nums):
        a=len(nums)//2
        for x in nums:
            c=nums.count(x)
            if c>a:
                return x
            else:
                return "False"
            break





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
