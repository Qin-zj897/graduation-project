n, m, k = map(int, input().split())

xd = m * (m-1) / 2 * n 
total =  xd * k * (k-1) / 2 
dis = xd * (k-1) * (k-2) / 2

ans = (total - dis) % (10e9+7)
print(ans)