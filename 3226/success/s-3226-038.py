def search(liebiao):
    for x in liebiao:
        jishu = 0
        for i in liebiao:
            if x == i:
                jishu = jishu +1 
        if jishu>len(liebiao)//2:
            return x
    return False






nums = eval(input())
y = search(nums)
print(y)


