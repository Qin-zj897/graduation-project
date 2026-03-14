def search(nums):
     for i in range(len(nums)):
          a = nums.count(nums[i])
          if a>=len(nums)/2:
              print(nums[i])
              break
          else:
                 print('False')
                 break





nums = eval(input())
y = search(nums)
print(y)


