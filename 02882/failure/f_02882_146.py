import math

def read():
    a, b, x = list(map(float, input().strip().split()))
    return a, b, x

def search_deg(a, b, s):
    l = 0.0
    r = 45.0
    for i in range(10000):
        m = (l + r) / 2
        if a * a * math.tan(math.radians(m)) / 2 > s:
            r = m
        else:
            l = m
    return l


def solve(a, b, x):
    s = x / a
    if x < a * a * b / 2:
        # 半分未満の時
        deg = 90.0 - search_deg(a, b, s)
    else:
        # 半分以上の時
        deg = search_deg(a, b, (a*b-s))
    return deg

if __name__ == '__main__':
    inputs = read()
    print("%.8f" % solve(*inputs))
