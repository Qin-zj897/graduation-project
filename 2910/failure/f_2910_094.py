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

    h,n=eval(read())
    r=0
    while n>1:
        r+=h*3/2
        h=h/2
        n-=1
        r+=h
    write("%.2f"%r)

    return ''.join(_outputs).rstrip('\n')


if __name__ == '__main__':
    input1 = eval(input())
    input2 = input()
    result = solve(input1, input2)
    if result != '':
        print(result)
