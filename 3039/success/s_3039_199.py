def solve(n):
    s=n.copy()
    max_num=max(n)
    min_num=min(n)
    for num in n:
        if num==max_num or num==min_num:
            s.remove(num)
    return s


if __name__ == '__main__':
    n = eval(input())
    result = solve(n)
    print(result)
