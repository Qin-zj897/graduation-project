def solve(nums):
    def search(nums):
        n=len(nums)

        for i in range(n):
            if nums.count(nums[i])>(n//2):
                a=nums[i]
                return(a)
        return("False")    





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
