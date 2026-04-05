def solve(h, N):
    lst=[h]
    for i in range(N-1):
        h=h/2
        lst.append(h)
    s=sum(lst)+sum(lst[1:-2])
    return "%.2f"%s


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
