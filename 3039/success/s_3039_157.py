def solve(Li):
    Imax = int(max(Li))
    Imin = int(min(Li))
    Nmax = Li.count(Imax)
    Nmin = Li.count(Imin)
    a=0
    b=0
    if Imax != Imin:
        while a < Nmax:
            Li.remove(Imax)
            a += 1
        while b < Nmin:
            Li.remove(Imin)
            b += 1
        return Li
    elif Imax == Imin:
        Li.clear()
        return Li


if __name__ == '__main__':
    Li = eval(input())
    result = solve(Li)
    print(result)
