def solve(nums):
    def search(arg):
        k = 0
        for i in range(len(arg)):
            a = arg.count(arg[i])
            if a >= len(arg) / 2:
                k = 1
                return arg[i]
            else:
                continue
        if k==0:
            return False
        elif k==1:
            pass






    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
