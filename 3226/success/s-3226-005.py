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





nums = eval(input())
y = search(nums)
print(y)


