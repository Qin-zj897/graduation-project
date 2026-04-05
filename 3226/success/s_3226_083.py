def solve(nums):
    def search(nums):
        n=len(nums)
        a=n//2
        b=0
        for i in nums:
            b=nums.count(i)
            if a<b:
              b=i
            else:
               b="False"
        return(b)





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
