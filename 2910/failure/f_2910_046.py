def solve(input1, input2):
    _inputs = [input1, input2]
    _idx = 0
    _outputs = []

    def read():
        nonlocal _idx
        value = _inputs[_idx]
        _idx += 1
        return value

    def write(*args, sep=' ', end='\n'):
        _outputs.append(sep.join(str(a) for a in args) + end)

    x=list(read())
    for i in range(len(x)):
        x[i]=(int(x[i])+5)%10
    x[0],x[1],x[-2],x[-1]=x[-1],x[-2],x[1],x[0]
    for i in x:
        write(i,end="")

    return ''.join(_outputs).rstrip('\n')


if __name__ == '__main__':
    input1 = list(input())
    input2 = input()
    result = solve(input1, input2)
    if result != '':
        print(result)
