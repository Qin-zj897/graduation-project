def solve(num):
    max=max(num)
    min=min(num)
    a=num.copy()
    for i in num:
        if i==max or i==min:
            a.remove(i)
    return a


if __name__ == '__main__':
    num = eval(input())
    result = solve(num)
    print(result)
