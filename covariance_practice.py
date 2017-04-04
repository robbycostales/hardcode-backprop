import numpy as np

# 1: mean , sd, var

d1 = [12, 23, 34, 44, 59, 70, 98]
d2 = [12, 15, 25, 27, 32, 88, 99]
print(np.mean(d1), end = "  ")
print(np.mean(d2))

print(np.std(d1), end = "  ")
print(np.std(d2))

print(np.var(d1), end = "  ")
print(np.var(d2))

# 2: cov

x = [10, 39, 19, 23, 28]
y = [43, 13, 32, 21, 20]

print(np.cov(x, y))

# 3: cov

D = [[1, -1, 4],
     [2, 1, 3],
     [1, 3, -1]]

print(np.cov(D))