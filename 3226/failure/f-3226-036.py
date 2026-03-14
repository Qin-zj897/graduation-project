def search(n):
    for i in n:
        if n.count(i) > len(n)//2:
            return i
        else:
            print("False")





nums = eval(input())
y = search(nums)
print(y)


