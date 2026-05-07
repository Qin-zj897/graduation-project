def main():
  a, b, x = map(int, input().split())
  vm = a*a*b
  if x>vm/2:
    return math.degrees(math.atan(2*(vm-x)/a**3))
  else:
    return math.degrees(math.atan(a*b*b/(2*x)))
if __name__ == '__main__':
  import math
  print(main())