def solve(a):
    max_value = max(a)
    min_value = min(a)
    while(max_value in a):
        a.remove(max(a))
    while(min_value in a):
        a.remove(min(a))
    return a


if __name__ == '__main__':
    a = eval(input())
    result = solve(a)
    print(result)
