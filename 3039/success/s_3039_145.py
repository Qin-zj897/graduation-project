def solve(s):
    max1=max(s)
    min1=min(s)
    nums=s.copy()
    for num in nums:
        if num ==max1 or num ==min1:
            s.remove(num)
    return s


if __name__ == '__main__':
    s = eval(input())
    result = solve(s)
    print(result)
