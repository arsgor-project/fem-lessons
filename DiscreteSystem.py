# from MATLAB codes for Finite Element Analysis
# problem1.m
# A.J.M. Ferreira, N. Fantuzzi 2019
# new line

import numpy as np 
import scipy.linalg as LA

# List of elements, connectivity
elementNodes = np.array([[0,1], [1,2], [1,3]])
# Number of elements
numberElements = len(elementNodes[:,0])
# Number of nodes
numberNodes = 4
# 
displacements = np.zeros(numberNodes)
force = np.zeros(numberNodes)
stiffness = np.zeros([numberNodes, numberNodes])
# Apply forces at the nodes
force[1] = 10.0

# Construct Stifness matrix
for i in range(numberElements):
    elementDof = elementNodes[i,:]
    stiffness[elementDof[0], elementDof] += np.array([[1.,-1.], [-1.,1.]])[:,0]
    stiffness[elementDof[1], elementDof] += np.array([[1.,-1.], [-1.,1.]])[:,1]


# Set BCs
allDof = np.arange(numberNodes)
prescribedDof = np.array([0,2, 3])
activeDof = np.setdiff1d(allDof, prescribedDof)

# Exclude prescribed DOFs, solve the system
A = stiffness[activeDof,:][:, activeDof]
F = force[activeDof]
displacements[activeDof] = LA.solve(A,F)

# Compute reactions
reactions = stiffness[prescribedDof, :] @ displacements

# Output displacements and reactions
print("Displacements: \n", displacements)
print("Reactions: \n", reactions)

