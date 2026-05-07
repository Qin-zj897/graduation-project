N, M, K = map(int, input().split())

MOD = 10**9+7

# WARN: Require mod number m is prime number
class ModComb:
    # For nCr mod m
    def __init__(self, n_max, m):
        self._n_max = n_max
        self._m = m
        self._mod_facts = [1] * (n_max+1)       # list of n! mod m
        self._mod_inv_facts = [1] * (n_max+1)   # list of inverse r! mod m

        if not m >= 3:
            raise ValueError('Require mod number m >= 3')

        # Calc n! mod m
        for i in range(2, n_max+1):
            self._mod_facts[i] = self._mod_facts[i-1] * i % m

        # Calc inverse r! mod m
        self._mod_inv_facts[n_max] = pow(self._mod_facts[n_max], m-2, m)
        for i in range(n_max-1, 1, -1):
            self._mod_inv_facts[i] = self._mod_inv_facts[i+1] * (i+1) % m

    def calc(self, n, r):
        # Return nCr mod m
        if n > self._n_max:
            raise Exception('n is larger than n_max')

        if r == 0:
            return 1

        m = self._m
        return self._mod_facts[n] * self._mod_inv_facts[r] * self._mod_inv_facts[n-r] % m

modcomb = ModComb(N*M-2, MOD)
p = modcomb.calc(N*M-2, K-2)

ans = 0
for i in range(N):
    for j in range(M):
        if i == 0 and j == 0:
            continue
        d = i+j
        cnt = (N-i) * (M-j)
        if i != 0 and j != 0:
            cnt *= 2
        ans += d * cnt % MOD
ans = ans * p % MOD

print(ans)
