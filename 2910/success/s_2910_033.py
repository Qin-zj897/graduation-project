def solve(h, n):
    kong = [h]
    if n==1:
        return h
    else:
        for i in range(n-1):
            h = h*0.5
            kong.append(h*2)
    return "%.2f"%(sum(kong))


if __name__ == '__main__':
    h = float(input())
    n = int(input())
    result = solve(h, n)
    print(result)
