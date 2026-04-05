def solve(nums):
    def search(n):
      f = "False"
      for x in n:
          cishu=n.count(x)
          zongshu=len(n)
          if cishu > zongshu//2:
            return x
      return f





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
