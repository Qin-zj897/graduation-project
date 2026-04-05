def solve(nums):
    def search(nums):
        n = len(nums)
        num_counts = {}
        for num in nums:
            if num in num_counts:
                num_counts[num] += 1
            else:
                num_counts[num] = 1
        for num, count in num_counts.items():
            if count > n // 2:
                return num
        return False





    y = search(nums)
    return y


if __name__ == '__main__':
    nums = eval(input())
    result = solve(nums)
    print(result)
