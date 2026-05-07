import math

rawUsrInput = input()
usrInput = rawUsrInput.split()

a = int(usrInput[0])
b = int(usrInput[1])
x = int(usrInput[2])

print(str((90 - (math.degrees(math.acos((b) / (math.sqrt((4 * x * x) / (a * a * b * b) + b * b))))))))