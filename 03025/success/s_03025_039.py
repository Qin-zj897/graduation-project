def prepare(n, MOD):
    facts = [1]
    t = 1
    for i in range(1, n + 1):
        t = t * i % MOD
        facts.append(t)
    invs = [1] * (n + 1)
    t = pow(t, MOD - 2, MOD)
    invs[n] = t
    for i in range(n, 1, -1):
        t = t * i % MOD
        invs[i - 1] = t
    return facts, invs


n, a, b, c = list(map(int, input().split()))
MOD = 10 ** 9 + 7
facts, invs = prepare(2 * n, MOD)

inv_ab = pow(a + b, MOD - 2, MOD)
a_ab = a * inv_ab % MOD
b_ab = b * inv_ab % MOD

win_a = 0
win_b = 0
exp_a = 1
exp_b = 1
for k in range(n):
    coef = facts[n + k] * invs[k] * invs[n - 1] % MOD
    win_a = (win_a + coef * exp_b) % MOD
    win_b = (win_b + coef * exp_a) % MOD
    exp_a = exp_a * a_ab % MOD
    exp_b = exp_b * b_ab % MOD

win_a = win_a * exp_a % MOD
win_b = win_b * exp_b % MOD

print((win_a + win_b) * 100 * inv_ab % MOD)
