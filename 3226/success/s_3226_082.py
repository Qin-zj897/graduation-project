def solve(nums):
    def search(ls):
        ls2=[]
        for i in ls:
            b=ls.count(i)
            if b>len(ls)/2:
                ls2.append(i)
                return(ls2[0])
                break    
        if ls2==[]:return(False)





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
