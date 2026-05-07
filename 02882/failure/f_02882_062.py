a,b,x = map(int,input().split())

S = a*b
I = x/a

if I > S/2:
  print(math.degrees(math.atan(2*(S-I)/a**2)))
else:
  print(90-math.degrees(math.atan(2*I/b**2)))
