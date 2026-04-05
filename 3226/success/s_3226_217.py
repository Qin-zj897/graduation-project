def solve(nums):
    def search(nums):
        for i in nums:
            a = int(nums.count(i))
            b = int(len(nums))//2
            if a>b:
                return i
            else:
                return False






    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
