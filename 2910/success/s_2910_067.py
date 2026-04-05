def solve(h, N):
    lst=[]
    a=h
    for i in range(N-1):
        a=a*0.5
        lst.append(2*a)
    H=sum(lst)+h
    return "%.2f"%H


if __name__ == '__main__':
    h = eval(input())
    N = eval(input())
    result = solve(h, N)
    print(result)
