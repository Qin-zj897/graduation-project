def solve(nums):
    def search(nums):
        n=len(nums)//2
        a=list(set(nums))
        lit=[]
        for i in a:
            b=nums.count(i)
            if b >n:
                lit.append(i)
        if lit==[]:
            return False
        else:
            return(max(lit))





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
