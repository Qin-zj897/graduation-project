from decimal import *
import math

def main():
    a, b, x = map(int, input().split())
    if Decimal(a) * Decimal(b) / Decimal(2) >= Decimal(x) / Decimal(a):
        tmp = Decimal(2) * Decimal(x) / (Decimal(a) * Decimal(b))
        c = math.atan(Decimal(tmp) / Decimal(b))
        d_deg = math.degrees(Decimal(c))
        print(Decimal(90) - Decimal(d_deg))
    else:
        d = (Decimal(2) * Decimal(x) / (Decimal(a) * Decimal(a))) - Decimal(b)
        c = math.atan(Decimal(a) / (Decimal(b) - Decimal(d)))
        d_deg = math.degrees(Decimal(c))
        print(Decimal(90) - Decimal(d_deg))


if __name__ == "__main__":
    main()