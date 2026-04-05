def solve(nums):
    def search(a):
        b=len(a)/2
        c=0
        for x in a:
            d=a.count(x)
            if d>b :
                c=x
            else:
                pass
        if c==0:
            return"False"
        else:
            return c





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
