def solve(lst):
    a = max(lst)
    b = min(lst)
    lst = [x for x in lst if x!= a and x != b]


    return lst


if __name__ == '__main__':
    lst = eval(input())
    result = solve(lst)
    print(result)
