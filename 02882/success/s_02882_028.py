def resolve():
    '''
    code here
    '''
    import math
    a, b, x = [int(item) for item in input().split()]

    s = x/a

    s_delta = a*b - s
    if s_delta <= a*b/2:
        print(math.degrees(math.atan((2* s_delta)/a**2)))
    else:
        print(math.degrees(math.atan(b**2/(2*s))))

if __name__ == "__main__":
    resolve()
