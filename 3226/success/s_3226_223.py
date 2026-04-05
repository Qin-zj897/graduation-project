def solve(nums):
    def search(nums):
        for i in nums:
            if nums.count(i)<len(nums)//2:
                 continue
            elif nums.count(i)>len(nums)//2:
                 return i
        return False






    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
