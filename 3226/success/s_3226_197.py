def solve(nums):
    def search(nums):
          n = len(nums)
          b = n//2
          for i in nums:
                if nums.count(i) > b:
                    c=i
                else:
                    c="False"
          return c





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
