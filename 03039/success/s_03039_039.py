import sys
def input(): return sys.stdin.readline().strip()
mod = 10**9+7

class Combination:
    """
    O(n)の前計算を1回行うことで，O(1)でnCr mod mを求められる
    n_max = 10**6のとき前処理は約950ms (PyPyなら約340ms, 10**7で約1800ms)
    使用例：
    comb = Combination(1000000)
    print(comb(5, 3))  # 10
    """
    def __init__(self, n_max, mod=10**9+7):
        self.mod = mod
        self.modinv = self.make_modinv_list(n_max)
        self.fac, self.facinv = self.make_factorial_list(n_max)

    def __call__(self, n, r):
        return self.fac[n] * self.facinv[r] % self.mod * self.facinv[n-r] % self.mod

    def make_factorial_list(self, n):
        # 階乗のリストと階乗のmod逆元のリストを返す O(n)
        # self.make_modinv_list()が先に実行されている必要がある
        fac = [1]
        facinv = [1]
        for i in range(1, n+1):
            fac.append(fac[i-1] * i % self.mod)
            facinv.append(facinv[i-1] * self.modinv[i] % self.mod)
        return fac, facinv

    def make_modinv_list(self, n):
        # 0からnまでのmod逆元のリストを返す O(n)
        modinv = [0] * (n+1)
        modinv[1] = 1
        for i in range(2, n+1):
            modinv[i] = self.mod - self.mod//i * modinv[self.mod%i] % self.mod
        return modinv

comb = Combination(2 * 100000)

def main():
    N, M, K = map(int, input().split())

    """
    xについてだけ考えればいいのはOK
    この後だが、xi, xjを固定したときに\Sum_{置き方} |xi - xj|を求めればよい。
    （シグマの順番を入れ替えた。つまりxiとxjに駒が置かれるような配置が何通りあるかを求めたい。）
    （iとjに騙されないこと！xi, xjと書くんじゃなくてx, x'と書いた方が分かりやすいかもしれない。）

    これは単純にcomb(NM-2, K-2)なので一応これで解けた。すなわち
        ans = \Sum_{xi, xj} |xi - xj| * comb(NM-2, K-2)

    しかしi, jに関してそれぞれforループを回しているのでこれだと間に合わない。
    まずcomb(NM-2, K-2)はi, jに依存しないので二重forループの外に出せる。
    そして\Sum_{i<=j}|xi-xj|だが、|xi-xj|=dとなる(i, j)の組は(N-d)*M^2個になる。（解説放送参照）
    """

    ans_x = 0
    for d in range(1, N):
        ans_x += d * (N - d) * M ** 2
        ans_x %= mod
    ans_y = 0
    for d in range(1, M):
        ans_y += d * (M - d) * N ** 2
        ans_y %= mod
    ans = (ans_x + ans_y) * comb(N * M - 2, K - 2)
    print(ans % mod)




if __name__ == "__main__":
    main()
