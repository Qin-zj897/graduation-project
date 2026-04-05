def solve(h, N):
    H = h
    if N == 1:
        return "%.2f"%h
    else:
        for i in range(1,N):
            H = H + 2*h*0.5**i
        return "%:.2f"%H


if __name__ == '__main__':
    h = float(input())
    N = int(input())
    result = solve(h, N)
    print(result)
