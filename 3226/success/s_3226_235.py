def solve(nums):
    def search(nums):
        for num in nums:
            if nums.count(num)>len(nums)//2:
                return num
            else:
                return False





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
