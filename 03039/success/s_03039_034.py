def solve():
    mod = 10 ** 9 + 7

    def cmb(n, r):
        def build():
            fs = [1]
            t = 1
            for x in range(1, h * w):
                t = (t * x) % mod
                fs.append(t)

            invs = [1] * (h * w)
            t = pow(t, mod - 2, mod)
            invs[-1] = t
            for x in range(h * w - 2, 0, -1):
                t = (t * (x + 1)) % mod
                invs[x] = t
            return fs, invs

        if r < 0 or r > n:
            return 0
        if r > (n - r):
            return cmb(n, n - r)
        if r == 0:
            return 1
        if r == 1:
            return n

        fs, invs = build()
        return fs[n] * invs[n - r] * invs[r] % mod

    h, w, k = map(int, input().split())
    c = cmb(h * w - 2, k - 2)
    ret = (pow(w, 2) * sum((h - d) * d for d in range(h))
           + pow(h, 2) * sum((w - d) * d for d in range(w))) * c % mod

    return ret


print(solve())