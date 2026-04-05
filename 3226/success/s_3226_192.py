def solve(nums):
    def search(nums):
        lit=[]
        for i in nums:
            a=nums.count(i)
            if a>(len(nums)//2):
                lit.append(str(i))
        if lit==[]:
            return False
        else:
            return(lit[0])





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
