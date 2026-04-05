def solve(Lst):
    a = max(Lst)
    b = min(Lst)
    while a in Lst:
        Lst.remove(a)
    while b in Lst:
        Lst.remove(b) 
    return Lst


if __name__ == '__main__':
    Lst = eval(input())
    result = solve(Lst)
    print(result)
