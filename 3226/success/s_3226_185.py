def solve(nums):
    def search(nums):
        ls=[]
        for i in nums:
            if nums.count(i)>len(nums)//2:
                ls.append(i)
                return i 
        if len(ls)==0:
                return False





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
