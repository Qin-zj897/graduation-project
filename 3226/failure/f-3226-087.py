#遍历列表，统计每一个元素相同的元素个数。
def search(liebiao):
    jishu = 0
    for x in liebiao:
        for i in liebiao:
            if x == i:
                jishu+=1
        if jishu>len(liebiao)//2:
            return x
    return False





nums = eval(input())
y = search(nums)
print(y)


