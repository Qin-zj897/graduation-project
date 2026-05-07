a,b,x = map(int,input().split())
from decimal import Decimal
if x>= (a**2)*b/2:
    tan = 2*(Decimal(a)*Decimal(a)*Decimal(b)-Decimal(x))/(Decimal(a)**3)
    import math
    answer = math.degrees(math.atan(tan))
else:
    pretan = 2*Decimal(x)/(Decimal(b)**2*Decimal(a))
    answer = Decimal(90)-Decimal(math.degrees(math.atan(pretan)))
print(answer)

