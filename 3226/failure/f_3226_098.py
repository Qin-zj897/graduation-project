def solve(nums):
    def search(n):

          for x in n:
              if n.count(x)>(len(n)//2):
                     return x
             else:
                     return False 





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
