def solve(nums):
    def search(nums):
        for i in range(len(nums)):
            a=nums.count(nums[i])
            if a>(len(nums)//2):
                return nums[i]
                break
            else:
                return False





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
