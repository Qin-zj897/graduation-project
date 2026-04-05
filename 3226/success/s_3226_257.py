def solve(nums):
    def search(nums):
        lst=[]
        for x in nums:
            if nums.count(x) > len(nums)//2:
               lst.append(x)
        if lst==[]:
            return "False"
        else:
            return x





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
