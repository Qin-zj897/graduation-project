import sys,math

sys.setrecursionlimit(10 ** 6)
int1 = lambda x: int(x) - 1
p2D = lambda x: print(*x, sep="\n")
def IS(): return sys.stdin.readline()[:-1]
def II(): return int(sys.stdin.readline())
def MI(): return map(int, sys.stdin.readline().split())
def LI(): return list(map(int, sys.stdin.readline().split()))
def LI1(): return list(map(int1, sys.stdin.readline().split()))
def LII(rows_number): return [II() for _ in range(rows_number)]
def LLI(rows_number): return [LI() for _ in range(rows_number)]
def LLI1(rows_number): return [LI1() for _ in range(rows_number)]

def main():
	a,b,x =MI()
	V = a*a*b
	if(V*0.5 >= x):
		t=2*x/(a*b)
		print(t)
		ans = 90 -math.degrees(math.atan(t/b))
	else:
		t=(2*x-(a**2)*b)/(a**2)
		t = b-t
		ans = math.degrees(math.atan(t/a))
	print(ans)
main()