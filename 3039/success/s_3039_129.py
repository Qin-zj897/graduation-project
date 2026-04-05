def solve(list_1):
    a=max(list_1)
    min=min(list_1)
    nums=list_1.copy()
    for i in nums:
       if i==a:
          list_1.remove(i)
       elif i==min:
          list_1.remove(i)
    return list_1


if __name__ == '__main__':
    list_1 = eval(input())
    result = solve(list_1)
    print(result)
