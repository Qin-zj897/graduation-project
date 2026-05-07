n, m, k = list(map(int, input().split()))

# 二項係数を 10^9+7 で割った余りを求める
# https://qiita.com/drken/items/3b4fdf0a78e7a138cd9a#5-%E4%BA%8C%E9%A0%85%E4%BF%82%E6%95%B0-ncr
# https://qiita.com/Yaruki00/items/fd1fc269ff7fe40d09a6
# https://www.hamayanhamayan.com/entry/2018/06/06/210256

mod = 10**9 + 7

# x ** a をmodで割った余りを、O(log(a))時間で求める。
def power(x, a):
	if a == 0:
		return 1
	elif a == 1:
		return x
	elif a % 2 == 0:
		return power(x, a//2) **2 % mod
	else:
		return power(x, a//2) **2 * x % mod

# xの逆元を求める。フェルマーの小定理より、 x の逆元は x ^ (mod - 2) に等しい。計算時間はO(log(mod))程度。
# https://qiita.com/Yaruki00/items/fd1fc269ff7fe40d09a6
def modinv(x):
	return power(x, mod-2)

# 二項係数の左側の数字の最大値を max_len　とする。nとかだと他の変数と被りそうなので。
# factori_table = [1, 1, 2, 6, 24, 120, ...] 要は factori_table[n] = n!
# 計算時間はO(max_len * log(mod))

max_len = n * m #適宜変更する

factori_table = [1] * (max_len + 1)
factori_inv_table = [1] * (max_len + 1)
for i in range(1, max_len + 1):
	factori_table[i] = factori_table[i-1] * (i) % mod

# 	factori_inv_table[i] = modinv(factori_table[i])
#	↑が遅いので、使うところだけ逆元を計算する↓
for i in [k-2, n*m-k]:
	factori_inv_table[i] = modinv(factori_table[i])

def binomial_coefficients(n, k):
	# n! / (k! * (n-k)! )
	return factori_table[n] * factori_inv_table[k] * factori_inv_table[n-k]

ans_1 = 0
for i in range(n):
	ans_1 += i * m**2 * (n-i) % mod
for i in range(m):
	ans_1 += i * n**2 * (m-i) % mod

print(binomial_coefficients(n*m-2, k-2) * ans_1 % mod)
