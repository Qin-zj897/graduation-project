def solve(nums):
    def search(list):
        a=0
        n=[]
        for i in list:
            if list.count(i)>a:
                a=list.count(i)
        b=len(list)/2
        if a>b:
            return(i)
        else:
            return(False)





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
