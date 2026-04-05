def solve(height, N):
    total_distance=height
    for _ in range(1,N):
        height*=0.5
        total_distance+=2*height
    return "%.2f"%total_distance


if __name__ == '__main__':
    height = float(input())
    N = int(input())
    result = solve(height, N)
    print(result)
