import math

def run():
  a, b, x = map(int, input().split())
  if a**2*b/2 > x:
    tan = 2*x/(a*b**2)
    print(90-math.degrees(math.atan(tan)))
  else:
    tan = 2*(a**2*b-x)/(a**3)
    print(math.degrees(math.atan(tan)))
  
if __name__ == '__main__':
  run()