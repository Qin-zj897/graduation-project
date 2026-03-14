def search(n):
      for x in n:
          if n.count(x)>(len(n)//2):
                 return x
      else:
                 return False 





nums = eval(input())
y = search(nums)
print(y)


