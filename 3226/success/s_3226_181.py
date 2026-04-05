def solve(nums):
    def search(nums):
        a=len(nums)//2
        for i in nums:
          b=nums.count(i)
        if b>a:
          return(i)
        else:
          return(False)






    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
