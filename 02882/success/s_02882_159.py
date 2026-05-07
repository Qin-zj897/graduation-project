import numpy as np

def main():
	A, B, X = [int(x) for x in input().split()]

	if (A * A * B) / 2.0 >= X:
		y = (2 * X) / (A * B)
		R = np.arctan(y / B)
		print(90 - np.rad2deg(R))
	else:
		_x = (A * A * B) - X
		y = (2 * _x) / (A * A)
		try:
			R = np.arctan(A / y)
			print(90 - np.rad2deg(R))
		except:
			print(0.000000000)
		
	
if __name__ == "__main__":
	main()
