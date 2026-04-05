def solve(nums):
    def search(x):
        jishu=0
        for x1 in x:
            if x.count(x1)>len(x)/2:
                jishu=x1
        if jishu!=0:
            return jishu
        elif jishu==0:
            return"False"





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
