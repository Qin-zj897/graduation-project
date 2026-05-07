import math

a,b,x=map(int,input().split())

if a == b:
  atan1 = 45
  
elif a > b:
  y = ( 2 * x / b ) /a
 
  if y <= a:
    ans = y/b
    atan1 = math.degrees(math.atan(ans))
    atan1 = 90 - atan1
    
  else :
    y = (2 * x / (a * a)) - b
    ans = a/(b-y)
    atan1 = math.degrees(math.atan(ans))
    atan1 = 90 - atan1
  
elif b > a:
  y = (2 * x / (a * a)) - b
  
  if y < 0:
    y = ( 2 * x / b ) /a
    ans = y/b
    atan1 = math.degrees(math.atan(ans))
    atan1 = 90 - atan1
    atan1 = 0
    
  else:
    ans = a/(b-y)
    atan1 = math.degrees(math.atan(ans))
    atan1 = 90 - atan1
  
  
print(atan1)