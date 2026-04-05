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

    n=read()
    m=1
    z=2
    h=0
    for x in range(n+1):
        m+=x
        z+=x
        h+=z/m
    write('%.2f'%h)

    return ''.join(_outputs).rstrip('\n')


if __name__ == '__main__':
    input1 = input()
    input2 = input()
    result = solve(input1, input2)
    if result != '':
        print(result)
