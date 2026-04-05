def solve(height, N):
    #一球从 h 米高度自由落下， 每次落地后反跳回原高度的 0.5倍；
    #  再落下， 求它在第 N 次落地时， 共经过多少米？
    if N == 1:
        d = height
    else :
        c = [height]
        for i in range(2,N+1) :
            b = height*((1/2)**(i-2))
            c.append(b)
        d = sum(c)
    return "%.2f"%(d)


if __name__ == '__main__':
    height = eval(input())
    N = eval(input())
    result = solve(height, N)
    print(result)
