def solve(nums):
    def search(nums):
        a=len(nums)//2
        b=0
        c=len(nums)
        for x in nums:
            b+=1
            if nums.count(x) > a:
                return x
            elif b == c:
                return "False"





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
