from __future__ import print_function
# import numpy as np
# import numpypy as np
import sys
input = sys.stdin.readline

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    return

import math
import string
import fractions
from fractions import Fraction
from fractions import gcd

def lcm(n,m):
    return int(n*m/gcd(n,m))

import re
import array
import copy
import functools
import operator

import collections
import itertools
import bisect
import heapq


from heapq import heappush
from heapq import heappop
from heapq import heappushpop
from heapq import heapify
from heapq import heapreplace

from queue import PriorityQueue as pq

# from itertools import accumulate
# from collections import deque
import random

def reduce(p, q):
    common = fractions.gcd(p, q)
    return (p//common , q//common )

def main():
    n,a,b,c = map(int, input().split())
    q=max(a,b)
    eprint("q: ",q)
    a=Fraction(a,100)
    b=Fraction(b,100)
    ###
    # exception=
    
    numerator=n*(a**n)*b + n*a*(b**n) +2*n*(a**n)*(b**n) - n*(a**n)*(b**2) -n*(a**2)*(b**n) -n*(a**2)*(b**n) - n*(a**(n+1))*(b**k) - k*(a**k)(b**(k+1))
    denominator=(1-a)(1-b)

    ##
    n=numerator
    q=denominator
    temp=reduce(n,q)
    n=temp[0]
    eprint("n:",n)
    q=temp[1]
    eprint("q: ",q)


    
    ##
    print(lcm(n,q))
    



    
    return

if __name__ == '__main__':
    main()
