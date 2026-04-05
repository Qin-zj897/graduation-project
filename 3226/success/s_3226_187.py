def solve(nums):
    def search(x):
        for i in x:
            c = x.count(i)
            n = len(x)
            nn = n//2
            if c > nn:
                return i
            else:
                return False  





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
