import math

    
def main():
    Input = [input() for i in range(1)]
    #N = int(Input[0])
    a,b,x = list(map(int,Input[0].split()))
    if x == a*a*b:
        print(0)
    else:
        a = float(a)
        b = float(b)
        x = float(x)
        ans = 90-math.atan(a/(2*(b-x/(a*a))))*180/math.pi
        if b * math.tan((90-ans)*math.pi/180) < a:
            ans = 90-math.atan(2*x/a/b/b)*180/math.pi
        print(ans)
        
if __name__ == "__main__":
    main()
    
