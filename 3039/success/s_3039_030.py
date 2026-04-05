def solve(lst):
    l2 = []
    for i in lst:
        if i == max(lst):
            continue
        elif i == min(lst):
            continue
        else:
            l2.append(i)        
    return l2


if __name__ == '__main__':
    lst = eval(input())
    result = solve(lst)
    print(result)
