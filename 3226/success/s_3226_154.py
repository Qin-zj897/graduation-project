def solve(nums):
    def search(nums):
        n=len(nums)
        b=[]
        for a in nums:
            if nums.count(a)>n//2:
                b.append(a)
            else:
                pass
        if b==[]:
            c="False"
        else:
            c=max(b)
        return c





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
