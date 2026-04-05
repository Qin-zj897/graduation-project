def solve(nums):
    def search(nums):
        a=0
        for x in nums:
            if nums.count(x)>a:
                a=nums.count(x)
                c=x
        b=len(nums)
        if a>b//2:
                h=c
        else:
                h=False
        return h





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
