# coding: utf-8
# Your code here!
import math
N, A, B, C = map(int,input().split())

A/=100
B/=100
C/=100

a = (N/(A/(A+B))+N)/2
b=0
if(b!=0):
    b=(N/(B/(A+B))+N)/2
    ans = math.ceil(a+b)//2
else:
    ans = math.ceil(a)
ans += math.ceil(C/N)

print(ans%(10**9+7))
