def solve(nums):
    def search(a):
        b=max(a,key=a.count)
        f=a.count(b)
        d=len(a)
        e=d//2
        if f>e:
            return b
        else :
            return "False"





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
