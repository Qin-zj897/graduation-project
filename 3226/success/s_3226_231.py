def solve(nums):
    def search(nums):
        n = len(nums)
        ls=[]
        a = False
        for x in nums:
            n1=nums.count(x)
            nums.remove(x)
            if n1 > n//2:
                a = x
                pass
        return a





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
