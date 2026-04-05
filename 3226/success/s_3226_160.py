def solve(nums):
    def search(nums):
        ls=[]
        a=len(nums)
        for i in nums:
            if nums.count(i)>a//2:
               ls.append(i)
        if len(ls)<1:
            y=False
        else:
            y=ls[0]
        return y





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
