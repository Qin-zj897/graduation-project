def solve(nums):
    def search(liebiao):
        for x in liebiao:
            jishu = 0
            for i in liebiao:
                if x == i:
                    jishu = jishu +1 
            if jishu>len(liebiao)//2:
                return x
        return False






    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
