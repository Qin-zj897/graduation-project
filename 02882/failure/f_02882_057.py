def resolve():
    import sys
    input = sys.stdin.readline
    # 整数 1 つ
    # n = int(input())
    # 整数複数個
    a, b, x = map(int, input().split())
    # 整数 N 個 (改行区切り)
    # N = [int(input()) for i in range(N)]
    # 整数 N 個 (スペース区切り)
    # N = list(map(int, input().split()))
    # 整数 (縦 H 横 W の行列)
    # A = [list(map(int, input().split())) for i in range(H)]

    import math
    import bisect
    #台形の場合
    if 2*x / (a**2) >= b:
        Down = [(x/(a**2) + a * math.tan(math.pi * i /(2* 9*10**6)) /2) for i in range(9*10**6)]
        ind = bisect.bisect_left(Down, b)

    # 三角形の場合
    else:
        Down = [math.sqrt(2*x /a * math.tan(math.pi * i /(2* 9*10**6))) for i in range(9*10**6)]
        ind = bisect.bisect_left(Down, b)

    print(float(ind/10**5))
resolve()