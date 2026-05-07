import math

def read():
    a, b, x = list(map(float, input().strip().split()))
    return a, b, x

def search_rad_overhalf(a, b, s):
    l = 0.0
    r = math.pi / 4.0
    while r - l > 1e-12:
        m = (l + r) / 2.0
        #if a * a * math.tan(m) > (a * b - s) * 2.0:
        if a * math.tan(m) > 2.0 * (b - s / a):
            r = m
        else:
            l = m
    return l

def search_rad_underhalf(a, b, s):
    l = 0.0
    r = math.pi / 4.0
    while r - l > 1e-12:
        m = (l + r) / 2.0
        if b * b * math.tan(m) > s * 2.0:
            r = m
        else:
            l = m
    return math.pi / 2.0 - l


def solve(a, b, x):
    s = x / a
    if x < a * a * b / 2.0:
        # 半分未満の時
        rad = math.atan2(b * b, 2.0 * s)
        #rad = search_rad_underhalf(a, b, s)
    else:
        # 半分以上の時
        #rad = math.atan2(2.0 * (b - s / a), a)
        rad = search_rad_overhalf(a, b, s)
    return math.degrees(rad)

if __name__ == '__main__':
    inputs = read()
    print("%.10f" % solve(*inputs))
