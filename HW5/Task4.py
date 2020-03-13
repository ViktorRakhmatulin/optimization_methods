# Import packages.
import cvxpy as cp
import numpy as np

# Generate a random non-trivial linear program.
profit = np.array([310,380,350,285])
weight_limit = ([10,16,8])
volume_limit = ([6800,8700,5300])
available_cargo = ([18,15,23,12])
volume_per_tonne = ([480,650,580,390])
# Define and solve the CVXPY problem.
X = cp.Variable((3,4))
function = np.ones(3)@(X@profit)
objective = cp.Maximize(function)

constraints =[X@np.ones(4) <= weight_limit]
constraints += [X@volume_per_tonne <= volume_limit]
constraints += [X.T@np.ones(3) <= available_cargo]
constraints += [X.flatten()[i]>=0 for i in range(12)]

prob = cp.Problem(objective,constraints)

prob.solve(verbose=True) 

# Print result.
print("\nThe optimal value is", prob.value)
print("A solution x is")
print(X.value)
print("A dual solution is")
print(prob.constraints[0].dual_value)
print(prob.constraints[1].dual_value)
print(prob.constraints[2].dual_value)
