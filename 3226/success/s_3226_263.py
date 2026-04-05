def solve(nums):
    def search(nums):
          n=len(nums)
          for i in nums:
               c=sum(1 for element in nums if element==i)
               if c>n/2:
                  return(i)
               else:
                  return('False')





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
