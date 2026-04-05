def solve(nums):
    def search(nums):
          for i in nums:
               b=nums.count(i)
               n=[]
               n.append(b)
          if max(n)>len(nums)/2:
               return(i)
          else: 
                return("False")





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
