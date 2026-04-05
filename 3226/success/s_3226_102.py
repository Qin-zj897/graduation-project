def solve(nums):
    def search(nums):
        dic={}
        n=len(nums)//2
        for x in nums:
            dic[x]=dic.get(x,0)+1
        maxv=max(dic.values())
        if maxv>n:
            for x in dic.keys():
                if dic[x]==maxv:
                    return x
        else:
            return 'False'





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
