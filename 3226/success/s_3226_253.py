def solve(nums):
    def search(a):
        for i in range(len(a)):
            if a.count(a[i])>(len(a)//2):
                c=a[i]
                return c
            return False






    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
