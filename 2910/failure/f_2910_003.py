def solve(h, N):
    H = [h,h*2]
    for i in range(1,N-1):
        H.append(int(H[i])+h/(2**i))
    return H[-1]


if __name__ == '__main__':
    h = float(input())
    N = int(input())
    result = solve(h, N)
    print(result)
