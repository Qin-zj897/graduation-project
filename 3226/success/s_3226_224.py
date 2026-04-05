def solve(nums):
    def search(nums):
        for i in nums:
            a = int(len(nums))
            if nums.count(i) > a//2 :
                return i
            else:
                return False





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
