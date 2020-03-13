import numpy as np
from random import randint
import scipy 
import matplotlib.pyplot as plt
import cvxpy as cp

def plot(name, values, label):
    _, ax = plt.subplots(1, figsize=(10, 6))
    x = np.linspace(0, 100, len(values))
    ax.plot(x, values, label=label)
    ax.set_title(name)
    ax.legend()
    plt.show()
def generate_laplas(n):
    L = np.zeros((n,n))
    for row in range(1,n,1):
        for col in range(row-1):
#            if (row==col): continue
            rand = -randint(0, 10)
            L[row, col] = rand
            L[col, row] = rand
            
    for inx in range (n):
        L[inx, inx] = abs(np.sum(L[:,inx]))
    return L
def gen_rand_vector(n):
    x = np.random.rand(n)
    for id in range(n):
        if (x[id]<0.5): 
            x[id] =-1
        else: x[id] = 1
    return x

def get_x_from_factor(v):
    x = np.zeros(n)
    eps = gen_rand_vector(n)
    eps /= np.linalg.norm(eps)
    for i in range(n):
        a =(v[:,i] @ eps)
        if (v[:,i] @ eps)>0: x[i] = 1
        else: x[i] = -1
    return x
        
n = 40
N = 1000
l = generate_laplas(n)
result = []
for attempt in range(N):
    x = gen_rand_vector(n)
    result.append (x.T @ l @x)
plot("naive", result, "naive")
naive = max(result)
naive_mean =  np.mean(result)
m = cp.Variable((n, n), PSD=True)
constraints =[m >> 0]
constraints += [ m[i][i] == 1 for i in range(n)]
obj = cp.Maximize(cp.trace(l@m))

prob = cp.Problem(obj, constraints)
prob.solve(eps = 1e-9)

res_from_sdp = m.value
#regularization
res_from_sdp +=1e-9 * np.eye(n)

V = scipy.linalg.cholesky(res_from_sdp, lower=False)
result.clear()
for attempt in range(N):
    x = get_x_from_factor(V)
    result.append (x.T @ l @x)

plot("SDP relaxation", result, "SDP relaxation")

print("The optimal value is", prob.value)
print ("naive", naive)
print ("top", max(result))
print("mean", np.mean(result))

from matplotlib.pyplot import cm
color=iter(cm.rainbow(np.linspace(0,1,6)))
width=4
plt.subplots(1, figsize=(10, 10))
c=next(color)
plt.axhline(prob.value, label = "SDP" , color = c,linewidth=width )
c=next(color)
plt.axhline(naive, label = "naive best result" , color = 'b',linewidth =width)
plt.axhline(naive_mean, label = "naive_mean" , color = c, ls ="dotted",linewidth=width)
c=next(color)
plt.axhline(max(result), label = "Goemans_Willamson" , color = c,linewidth=width)
c=next(color)
plt.axhline(np.mean(result), label = "Goemans_Willamson mean" , color = c, ls ="dotted",linewidth=width)
plt.title("comparisson")
plt.legend()
plt.show()

