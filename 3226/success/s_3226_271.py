def solve(nums):
    def search(nums):
      n=len(nums)
      dic={}
      for i in nums:
        if i not in dic:
          dic[i]=1
        if i in dic:
          dic[i]+=1
      for s in dic.keys():
        if dic[s]>(n//2+1):
          return s
      else:
        return False








    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
