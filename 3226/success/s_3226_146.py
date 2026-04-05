def solve(nums):
    def search(x):
        m=1
        a=len(x)
        b=a//2
        for i in x:
            c=x.count(i)
            if c>b:
                return (i)
                m=0
                break
            if m:
                return ('False')





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
