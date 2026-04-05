def solve(ls):
    max_num = max(ls)
    min_num = min(ls)
    nums = ls.copy()
    for num in nums:
        if num == max_num or num == min_num:
            ls.remove(num)
    return ls


if __name__ == '__main__':
    ls = eval(input())
    result = solve(ls)
    print(result)
