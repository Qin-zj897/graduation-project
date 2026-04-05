def solve(nums):
    def search(nums):
        for n in nums:
            if nums.count(n)>len(nums)//2:
                return n
            else:
                return False





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
