def solve(nums):
    def search(nums):
        ls = []
        n = len(nums)
        for x in nums:
            if nums.count(x) > n/2:
                a = x
                ls.append(x)
            else:
                pass
        if ls==[] :
            a = "False"
        else:
            pass
        return(a)





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
