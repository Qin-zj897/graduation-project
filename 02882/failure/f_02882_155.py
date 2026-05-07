import math
a,b,x = map(int, input().split())
def f(a,b,t):
  if t < math.atan(a/b): return a*a*b - a*a*a*math.tan(t)/2
  elif t > math.atan(a/b): return a*b*b*math.tan(math.pi/2-t)/2
  else: return a*a*b/2
  
th = [math.atan(a/b), 0, math.pi/2]
err = float("inf")
eps = 1e-6
while abs(err) > eps:
  err = x - f(a,b,th[0])
  if abs(err) < eps: break
  elif err < 0:
    th[0], th[1] = (th[2]-th[0])/2+th[0], th[0]
  elif err > 0:
    th[0], th[2] = (th[0]-th[1])/2+th[1], th[0]
    
  #print(err, th)
print(math.degrees(th[0]))