def solve(nums):
    def search(nums):
        n=len(nums)//2
        ls=[]
        for i in nums:
            d=0
            for x in nums:
                if x==i:
                    d+=1
            if d>n:
                ls.append(i) 
        if bool(ls)==False:
            x="False"
        else:
            x=ls[0]
        return x





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
