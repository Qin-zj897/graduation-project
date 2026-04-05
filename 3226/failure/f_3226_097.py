def solve(nums):
    def search(n):
        list=[]
        for x in n:
            b=n.count(x)
            list.append(b)
        c=max(list)    
        if c>(len(n)//2):
            return c
        else:
            return False 





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
