def solve(a):
    for i in range(0,2):
        a.remove(max(a))
        a.remove(min(a))
    return a


if __name__ == '__main__':
    a = list(eval(input()))
    result = solve(a)
    print(result)
