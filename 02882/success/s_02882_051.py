import math
def main():
    a, b, x = map(float, input().split())

    # 底面三角形の高さ
    h = 2*x/(a*b)

    if h > a:
        res = a*a*b-x
        j = res*2/(a*a)
        rad = math.degrees(math.atan2(j,a))
    else:
        rad = math.degrees(math.atan2(b,h))
    #max_cos = (2*x)/(b*h*a)
    #rad = math.asin(max_cos)

    print(rad)

if __name__ == '__main__':
    main()

