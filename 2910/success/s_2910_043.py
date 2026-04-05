def solve(height, N):
    sum=height
    for x in range(N-1):
        sum+=height
        height=height/2
    return "%.2f"%sum


if __name__ == '__main__':
    height = eval(input())
    N = int(input())
    result = solve(height, N)
    print(result)
