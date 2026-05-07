LAW = 10 ** 9 + 7

N, M, K = map(int, input().split(" "))

def calc_rev(n, law):
	if n == 1:
		return 1
	
	pre_r = law
	pre_x = 0
	pre_y = 1
	cur_r = n
	cur_x = 1
	cur_y = 0
	
	while True:
		d = pre_r // cur_r
		r = pre_r % cur_r
		
		nxt_r = r
		nxt_x = pre_x - d * cur_x
		nxt_y = pre_y - d * cur_y
		
		if nxt_r == 1:
			ret = nxt_x
			while ret < 0:
				ret += law;
			return ret;
		elif nxt_r == 0:
			return None
		
		pre_r = cur_r;
		pre_x = cur_x;
		pre_y = cur_y;
		cur_r = nxt_r;
		cur_x = nxt_x;
		cur_y = nxt_y;

def calc_dsum(n, m):
	dx = 0
	for i in range(1, n):
		dx += i * (n - i)
	
	dy = 0
	for i in range(1, m):
		dy += i * (m - i)
	dx *= m * m
	dy *= n * n
	
	return dx + dy

def calc_csum(n, k):
	csum = 1
	for i in range(k):
		csum = (csum * (n - i)) % LAW
	for i in range(1, k + 1):
		csum = (csum * calc_rev(i, LAW)) % LAW
	
#	print(csum)
	
	csum = (csum * k * (k - 1)) % LAW
	csum = (csum * calc_rev(n * (n - 1), LAW)) % LAW
	
	return csum

def c(n, m, k):
	
	return (calc_csum(n * m, k) * calc_dsum(n, m)) % LAW

print(c(N, M, K))
