def solve(a, b):
    def calculate(a, b):
        d = a
        if b > 1:
            for x in range(1, b):
                c = a * (1/2)**(x)
                d = d + c * 2
            return round(d, 2)
        else:
            return round(d, 2)

    if __name__ == "__main__":
        result = calculate(a, b)
        return "%.2f" % result


if __name__ == '__main__':
    a = float(input())
    b = int(input())
    result = solve(a, b)
    print(result)
