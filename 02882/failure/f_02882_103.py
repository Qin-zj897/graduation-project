import math
input01 = input()
a=input01.split()

tate=float(a[0])
takasa=float(a[1])
mizu=float(a[2])
menseki1 = tate * takasa
menseki2 = mizu / tate
diff=menseki1-menseki2
rad=0
if(diff <= tate*takasa/2):
    stakasa=2*diff/(tate)
    rad = math.degrees(math.atan(stakasa/tate))
elif(diff > tate*takasa/2):
    stakasa=2*(tate*takasa-diff)/(tate)
    rad = math.degrees(math.atan(tate/stakasa
                                 )) 
    

print(rad)