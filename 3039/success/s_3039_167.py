def solve(ls1):

    ls2 = []

    for i in ls1:

        if i==max(ls1):

            continue

        elif i==min(ls1):

            continue

        else:

            ls2.append(i)

    return ls2


if __name__ == '__main__':
    ls1 = eval(input())
    result = solve(ls1)
    print(result)
