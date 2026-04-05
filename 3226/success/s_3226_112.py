def solve(nums):
    def search(nums):
        a = len(nums)
        for i in nums:
            b = nums.count(i)
            c = 0
            if b> a/2:
                c = 1
                return i
        if c==0:
            return("False")








    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
