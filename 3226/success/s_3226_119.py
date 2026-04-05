def solve(nums):
    def search(n):
        num=0
        an=0
        for x in n:
            if n.count(x)>len(n)/2:
                num+=1
                an=x
        if num==0:
            an='False'
        return an





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
