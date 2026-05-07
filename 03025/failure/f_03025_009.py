import sys

sys.setrecursionlimit(110000)


def inpl():
    return list(map(int, input().split()))


class UnionFind:
    def __init__(self, N):
        # par = parent
        self.par = [i for i in range(N)]
        self.N = N
        self.hop = {i: -1 for i in range(N)}
        self.adj = [[] for i in range(N)]
        return

    def root(self, x):
        if self.par[x] == x:
            return (x)
        else:
            self.par[x] = self.root(self.par[x])
            return self.par[x]

    def same(self, x, y):
        return (self.root(x) == self.root(y))

    def union(self, x, y):
        x = self.root(x)
        y = self.root(y)
        if (x == y):
            return False

        self.par[x] = min(x, y)
        self.par[y] = min(x, y)
        return True

    def find_root(self):
        for i in range(self.N):
            if i == self.root(i):
                return i
        return

    def dfs(self, i):
        for nh in self.adj[i]:
            if self.hop[nh] != -1:
                continue
            else:
                self.hop[nh] = self.hop[i] + 1
                self.dfs(nh)
        return


N = int(input())
uf = UnionFind(N)
AB = [[0, 0] for i in range(N - 1)]

for i in range(N - 1):
    a, b = inpl()
    a -= 1
    b -= 1
    AB[i][0] = a
    AB[i][1] = b
    uf.adj[a].append(b)
    uf.adj[b].append(a)

C = inpl()

for a, b in AB:
    uf.union(a, b)

root_num = uf.find_root()
uf.hop[root_num] = 0
uf.dfs(root_num)

dic2 = sorted(uf.hop.items(), key=lambda x: x[1], reverse=True)

ans = [0 for i in range(N)]
for i, k in enumerate(dic2):
    ans[k[0]] = i

C.sort()

print(sum(C[:-1]))
for i in range(N):
    if i != 0:
        print(" ", end="")
    print(C[ans[i]], end="")
print("")
