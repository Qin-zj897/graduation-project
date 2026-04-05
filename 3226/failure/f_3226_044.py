def solve(nums):
    def search(nums):
        for x in nums:
            if nums.count(x) > 1:
                a = x
        if a :
            a = "False"
        return(a)





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
