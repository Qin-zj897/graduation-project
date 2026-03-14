def search(nums):
    count = 0  # 初始化计数器
    candidate = None  # 初始化候选元素

    for num in nums:
        if count == 0:
            candidate = num
        if num == candidate:
            count += 1
        else:
            count -= 1

    # 验证候选元素是否为多数元素
    count = nums.count(candidate)
    if count > len(nums) // 2:
        return candidate
    else:
        return False





nums = eval(input())
y = search(nums)
print(y)


