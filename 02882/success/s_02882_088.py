import numpy as np
A = list(map(int, input().split()))
under = A[0] ** 2
total = under * A[1]
if A[2] >= total // 2:
  b = (total - A[2]) * 2 / under
  print(90 * np.arctan(b / A[0]) / np.pi * 2)
else:
  under = A[2] * 2 / A[1]
  a = under / A[0]
  print(90 * np.arctan(A[1] / a) / np.pi * 2)
  