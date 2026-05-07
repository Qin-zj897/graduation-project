a, b, x = map(int, input().split())

dig = (1 - ((x / (a * a)) / b)) * 90
print(dig)