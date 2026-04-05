def solve(nums):
    def search(n):
        b=[]
        for x in n:
            if n.count(x)>len(n)/2:
                c=x
            else:
                c=False
        return c





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
