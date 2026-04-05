def solve(nums):
    def search(nums):
        for x in nums:
            a=nums.count(x)
            if a>len(nums)/2:
                return x
                break
        else:
            return False





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
