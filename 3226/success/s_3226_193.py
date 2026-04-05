def solve(nums):
    def search(nums):
        lit=''
        lit1=list(set(nums))
        for i in lit1:
            a=nums.count(i)
            if a>(len(nums)//2):
                lit+=str(i)
        if lit=='':
            return False
        else:
            return(lit)





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
