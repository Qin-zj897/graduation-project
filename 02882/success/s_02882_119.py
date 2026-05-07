import numpy as np
def solve(a, b, x):
    if a < b:
        if x < a*a*b/2:
            h = 2*x / (a*b)
            theta = np.arctan(b/h)
        else:
            h = 2*(a*a*b - x) / (a*a)
            theta = np.arctan(h/a)
    else:
        if x < a*a*b/2:
            h = 2*x / (a*b)
            theta = np.arctan(b/h)
        else:
            h = 2*(a*a*b - x) / (a*a)
            theta = np.arctan(h/a)
    return theta / np.pi * 180

a, b, x = map(int, input().split())
print(solve(a, b, x))