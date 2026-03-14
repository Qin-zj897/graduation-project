def search(n):
    a = n.count(max(n,key = n.count))
    b = max(n,key = n.count)
    if a > len(n)//2:
        return b
    else:
        return 'False'





nums = eval(input())
y = search(nums)
print(y)


