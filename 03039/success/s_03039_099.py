

class Cmber:
    def __init__(self,gensize,mod=10**9+7):
        self.mod = mod
        self.g1=[1,1]
        self.g2 = [1, 1] #逆元テーブル
        self.inverse = [0, 1] #逆元テーブル計算用テーブル
        self.getgen(gensize)
    def getgen(self,N):
        for i in range( 2, N + 1 ):
            self.g1.append( ( self.g1[-1] * i ) % self.mod )
            self.inverse.append( ( -self.inverse[self.mod % i] * (self.mod//i) ) % self.mod )
            self.g2.append( (self.g2[-1] * self.inverse[-1]) % self.mod )
 
    def cmb(self,n, r):
        if ( r<0 or r>n ):
            return 0
        r = min(r, n-r)
        return self.g1[n] * self.g2[r] * self.g2[n-r] % self.mod
 
    def cmb_very_high_n(self,n, r):
        if ( r<0 or r>n ):
            return 0
        g1=1
        r = min(r, n-r)
        for i in range(n-r+1,n+1):
            g1*=i
            g1%=self.mod
        return g1 * self.g2[r] % self.mod

    def multiple_cmb(self,vals):
        ans = self.g1[sum(vals)] % self.mod
        for i in vals:
            ans = (ans*self.g2[i])%self.mod
        return ans










def resolve():
    N,M,K = map(int,input().split())
    cmber = Cmber(N*M)
    score = 0
    for dist in range(1,N):
        cmb_of_2 = dist*(((((N-dist)*M)%cmber.mod)*M)% cmber.mod) % cmber.mod

        score = (score+(cmb_of_2*cmber.cmb(N*M-2,K-2))%cmber.mod)%cmber.mod
        
    for dist in range(1,M):
        cmb_of_2 = dist*(((((M-dist)*N)%cmber.mod)*N)% cmber.mod) % cmber.mod

        score = (score+(cmb_of_2*cmber.cmb(N*M-2,K-2))%cmber.mod)%cmber.mod

    print(score)

if __name__ == "__main__":
    resolve()