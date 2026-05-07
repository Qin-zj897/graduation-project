import sys, math, collections, heapq, itertools
F = sys.stdin
def single_input(): return F.readline().strip("\n")
def line_input(): return F.readline().strip("\n").split()
mod = 7 + 10 ** 9
factorial = [1] * (2 * 10 ** 5 + 1)
for i in range(1, 2 * 10 ** 5 + 1):
    factorial[i] = (factorial[i-1] * i) % mod
revfactorial = [1] * (2 * 10 ** 5 + 1)
revfactorial[-1] = pow(factorial[-1], mod - 2, mod)
for i in reversed(range(1, 2 * 10**5)):
    revfactorial[i] = (revfactorial[i+1] * (i+1)) % mod

def comb(n, r):
    return (factorial[n] * revfactorial[n-r] * revfactorial[r]) % mod

def solve():
    N, M, K = map(int, line_input())
    xsum, ysum = 0, 0
    for i in range(1, N): #xsum. (pairの取り方)  * d
        xsum += ((N - i) * M * M * i) % mod
        xsum %= mod
    for j in range(1, M):
        ysum += ((M - j) * N * N * j) % mod
        ysum %= mod
    print(((xsum + ysum) * comb(N*M-2, K-2)) % mod)
    return 0
  
if __name__ == "__main__":
    solve()