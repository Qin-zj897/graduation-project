def solve(nums):
    def search(nums):
        c=0
        for i in nums:
            if nums.count(i)>len(nums)//2:
                c=1
                return i
                break
        if c==0:
            return "False"





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
