import math
n, a, b, c = map(int,input().split())
prob=(a)/100.0
answer = math.floor(1/(1-prob))

print(answer)