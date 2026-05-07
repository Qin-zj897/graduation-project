import math

a, b, x = list(map(int, input().split()))

tmp = (2*a**2*b-2*x)/a**3
theta0 = math.degrees(math.atan(tmp))

aa = (2*b-a*math.tan(math.radians(theta0)))*a**2/2

theta1 = math.degrees(math.atan(2*x/a**3))
bb = a**3*math.tan(math.radians(theta1))


theta2 = math.degrees(math.atan(a*b**2/(2*x)))
cc = a*b**2/(2*math.tan(math.radians(theta2)))

diff = [(abs(x-aa), theta0), (abs(x-bb), theta1), (abs(x-cc), theta2)]
diff = sorted(diff)
print(diff[0][1])