def solve(a):
    a.remove(max(a))
    a.remove(min(a))
    return a


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
