S = input().split(' ')
n = int(S[0])
a = int(S[1])
b = int(S[2])
c = int(S[3])
win = a if a>=b else b

print(int(n*100/win))