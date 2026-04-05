def solve(nums):
    def search(ls):
        a=len(ls)//2
        kong=[]
        for x in ls:
            b=ls.count(x)
        if  b>a:
            kong.append(x)
            return x
        if b<=a:
            return False





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
