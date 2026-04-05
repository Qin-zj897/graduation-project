def solve(nums):
    ls = []
    def search(nums):
        n = len(nums)
        for x in nums:
            if nums.count(x) > n/2:
                ls.append(x)
                return x
            else:
                pass
        return(nums)
    if ls == []:
        return "False"
    else:
        pass





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
