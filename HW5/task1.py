import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.sparse
import cvxpy as cp

def plot(names, xs, ys, labels):
    _, ax = plt.subplots(1, 3, figsize=(20, 3))
    for idx, (x_, y_, name, label) in enumerate(zip(xs, ys, names, labels)):
        ax[idx].plot(x_, y_, label=label)
        ax[idx].set_title(name)
        ax[idx].legend()
    plt.show()

def min_with_penalty(gamma,matrix, norm):
    function_values = []
    for interpolator in interpolating_functions:
        f = cp.Variable(n)
        y_ = interpolator(x_) + noise
        function = cp.norm(f - y_, p=2) + gamma*cp.norm(matrix * f, p=norm)
        problem = cp.Problem(cp.Minimize(function))
        problem.solve()
        function_values += [f.value]
    return function_values

class piece_wize_lin_discontinious:
    def __init__(self, x, y, sigma=1.):
        n = len(x)
        dy = np.random.normal(0, sigma, n)
        self.y = np.stack([y + dy, y]).flatten('F')
        self.x = np.stack([x, x]).flatten('F')
        self.interp = interp1d(self.x, self.y, kind='linear')
    
    def __call__(self, y):
        return self.interp(y)

# create the functions
n = 500
k = 8
x = np.linspace(0, 1, k)
y = np.random.rand(k)
pwlc = interp1d(x, y, kind='linear')
pwld = piece_wize_lin_discontinious(x, y, sigma=0.3)
quadspline = interp1d(x, y, kind='quadratic')

interpolating_functions = [quadspline,pwlc, pwld]
names =  ["quadratic spline","piece-wise linear continuous", "piece-wise linear discontinuous"]
x_ = np.linspace(0, 1, n)
noise = np.random.normal(0, 0.05, n)

D1 = scipy.sparse.diags([1, -1], [0, 1], shape=(n-1, n)).toarray()
D2 = scipy.sparse.diags([1, -2, 1], [0, 1, 2], shape=(n-2, n)).toarray()

plot(names, [x_] * 3, [interpolator(x_) for interpolator in interpolating_functions], [r"$f(x)$"] * 3)
plot(names, [x_] * 3, [interpolator(x_) + noise for interpolator in interpolating_functions], [r"$f(x) + noise$"] * 3)

gamma = 0.5
for inx in range(4):
    reconst_values = min_with_penalty(gamma,D1,2)
    plot(names, [x_] * 3, reconst_values, [r"penalty " +str(gamma) + " * $||D_1\hat{x}||_2$"] * 3)
    
    reconst_values = min_with_penalty(gamma,D1,1)
    plot(names, [x_] * 3, reconst_values, [r"penalty " +str(gamma) + " * $||D_1\hat{x}||_1$"] * 3)
    
    reconst_values = min_with_penalty(gamma,D2,1)
    plot(names, [x_] * 3, reconst_values, [r"penalty " +str(gamma) + " * $||D_2\hat{x}||_1$"] * 3)
    gamma+=2.5
