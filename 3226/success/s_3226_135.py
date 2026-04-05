def solve(nums):
    def search(nums):
     n = 0
     for num in nums:
      if nums.count(num) > (len(nums)//2):
       n = 1
       return num
      if n == 0:
       return  False







    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
