def solve(height, N):
    sum=0
    for x in range(N+1):
        sum+=height/2
        height=height/2
    return "%.2f"%sum


if __name__ == '__main__':
    height = eval(input())
    N = int(input())
    result = solve(height, N)
    print(result)
