def search(n):
    for i in n:
        if n.count(i) >len(n)//2:
            return n.count(i)
        else:
            print('False')





nums = eval(input())
y = search(nums)
print(y)


