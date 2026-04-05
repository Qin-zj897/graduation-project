def solve(lis):
    num_max = lis.count(max(lis))
    num_min = lis.count(min(lis))
    for i in range(num_max):
        lis.remove(max(lis))
    if lis != []:
        for b in range(num_min):
            lis.remove(min(lis))
    return lis


if __name__ == '__main__':
    lis = eval(input())
    result = solve(lis)
    print(result)
