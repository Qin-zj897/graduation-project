def solve(nums):
    def search(a):
    n = len(a)
        for x in a:
            if a.count(x)>n//2:
                m = x
            else:
                m=False
        return m





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
