def solve(nums):
    def search(nums):
        a=0
        le=len(nums)/2
        for x in nums:
            if nums.count(x)>le:
                a=x
                if  a!=0:
                    return a
                elif a==0:
                    return "False"






    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
