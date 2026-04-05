def solve(nums):
    def search(nums):
        b=[]
        for i in nums:
            b.append(nums.count(i))
            if max(b)>len(nums)//2:
                return(i)
            else:
                return('False')





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
