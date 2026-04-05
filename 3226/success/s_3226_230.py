def solve(nums):
    def search(x):
        a=len(x)
        b=0
        for i in x:
            if x.count(i)>a//2:
                b=i
        if b!=0:
            return b
        elif b==0:
            return "False"





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
