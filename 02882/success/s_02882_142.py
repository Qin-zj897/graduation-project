import math

abc = input()
abc_list = abc.split(' ')
abc_intlist = []
for c1 in abc_list:
  abc_intlist.append(int(c1))
a = abc_intlist[0]
b = abc_intlist[1]
x = abc_intlist[2]
if (a**2)*b<2*x:
    print(math.degrees(math.atan(  2*(b-x/(a**2))/a  )  ))
else:
    print(math.degrees(math.atan(  a*(b**2)/(2*x)  )  ))
