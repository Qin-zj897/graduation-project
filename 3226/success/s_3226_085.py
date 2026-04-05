def solve(nums):
    def search(nums):
           n = int(len(nums)/2)
           for i in nums:
                 if nums.count(i)>n:
                      h = i
                 else:
                      h = "False"
           return  h





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
