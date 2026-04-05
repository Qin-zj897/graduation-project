def solve(a):
    max_num=max(a)
    min_num=min(a)
    for i in a:
        if i==max_num or i==min_num:
            a.remove(i)
    return a


if __name__ == '__main__':
    a = list(eval(input()))
    result = solve(a)
    print(result)
