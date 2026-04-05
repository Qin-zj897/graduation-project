def solve(h, a):
    t=h*(3-2*(0.5)**(a-1))
    return "{:.2f}".format(t)


if __name__ == '__main__':
    h = eval(input())
    a = eval(input())
    result = solve(h, a)
    print(result)
