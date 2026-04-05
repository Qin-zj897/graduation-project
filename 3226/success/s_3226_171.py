def solve(nums):
    def search(ls):
        for i in ls:
            if ls.count(i) > len(ls)//2:
                return i
            else:
                return 'False'





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
