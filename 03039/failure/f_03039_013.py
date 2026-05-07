import sys
def LI(): return [int(x) for x in sys.stdin.readline().split()]
def II(): return int(sys.stdin.readline())
def LS(): return sys.stdin.readline().split()
sys.setrecursionlimit(10**7)
INF = 10 ** 18
MOD = 10 ** 9 + 7
def LI_(): return [int(x) - 1 for x in sys.stdin.readline().split()]
def LF(): return [float(x) for x in sys.stdin.readline().split()]
def SI(): return input()
YN = lambda b: print('YES') if b else print('NO')
yn = lambda b: print('Yes') if b else print('No')

from scipy.special import comb

def main():
    N, M, K = LI()

    rows = M*M*sum([d * (N-d) for d in range(1, N)])
    cols = N*N*sum([d * (M-d) for d in range(1, M)])
    ans = (rows + cols) * comb(M*N-2, K-2, exact=True) % MOD

    print(ans)

main()