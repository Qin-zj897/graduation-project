def solve(nums):
    #遍历列表，统计每一个元素相同的元素个数。
    def search(liebiao):

        for x in liebiao:
            jishu = 0
            for i in liebiao:
                if x == i:
                    jishu+=1
            if jishu>len(liebiao)//2:
                return x
        return False
    y  =  search(nums)
    return y
if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
