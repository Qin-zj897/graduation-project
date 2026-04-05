def solve(nums):
    def search(nums):
        dic={}
        for i in nums:
            if i not in search:
                search[i]=1
            else:
                search[i]+=1
        return max(search.values())






    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
