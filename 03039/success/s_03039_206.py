#!/usr/bin/python3
# -*- coding: utf-8 -*-


MOD = 10**9+7
Facts = [1]*(2*10**5+1)
for i in range(2,2*10**5+1):
  Facts[i] = (Facts[i-1]*i) % MOD

def mul(a,b):
  return (a*b) % MOD

def power(a,b):
  if   b==0   : return 1
  elif b==1   : return a % MOD
  elif b%2==0 : return power(a,b//2)**2 % MOD
  else        : return power(a,b//2)**2 * a % MOD

def div(a,b):
  return mul(a,power(b,MOD-2))


def combi(n,m,k):
  if k <= 2 or k >= n*m:
    return 1
  else:
    return div(Facts[n*m-2],(Facts[k-2]*Facts[n*m-k]))


def main():
  N,M,K = map(int, input().split())
  com = combi(N,M,K)
  ans = 0
  for d in range(1,N):
    ans = (ans + d * (N-d)*M**2 * com) % MOD
  for d in range(1,M):
    ans = (ans + d * (M-d)*N**2 * com) % MOD
  return ans

if __name__ == "__main__":
  print(main())