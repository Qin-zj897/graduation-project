def solve(h, n):
    from os import times



    def bounce(s = h, height = h, times = 1):
        return s if times ==n else bounce(s + height, height/2, times + 1)

    return f"{bounce():.2f}"


if __name__ == '__main__':
    h = float(input())
    n = float(input())
    result = solve(h, n)
    print(result)
