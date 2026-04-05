def solve(H, N):
    s=-H
    for i in range(N):
       s+=H*2
       H*=(0.5)
    return '%.2f'%s


if __name__ == '__main__':
    H = eval(input())
    N = eval(input())
    result = solve(H, N)
    print(result)
