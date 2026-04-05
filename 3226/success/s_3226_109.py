def solve(nums):
    def search(ls):
        n = len(ls)
        for x in ls:
            if ls.count(x) > n//2:
                return x
                break
            else:
                return "False"





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
