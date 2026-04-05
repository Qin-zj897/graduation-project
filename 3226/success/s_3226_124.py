def solve(nums):
    def search(nums):
        i=0
        nums1=list(set(nums))
        for x in nums1:
            if nums.count(x)> len(nums)//2:
                return x

        if i==0:
            return False





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
