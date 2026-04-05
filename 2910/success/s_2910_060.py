def solve(h, n):
    #一球从 h 米高度自由落下， 每次落地后反跳回原高度的 0.5倍； 再落下， 求它在第 N 次落地时， 共经过多少米？  
    #【输入形式】第一行输入高度，第二行输入N
    sums=h
    for x in range(n-1):
        sums+=h*(0.5)**(x)
    return "%.2f"%(sums)
    #第一次落地 h 第二次落地h + H*0.5**n-1*2


if __name__ == '__main__':
    h = int(input())
    n = int(input())
    result = solve(h, n)
    print(result)
