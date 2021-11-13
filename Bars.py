# from MATLAB codes for Finite Element Analysis
# problem2.m
# A.J.M. Ferreira, N. Fantuzzi 2019
# %% [markdown]
# <h1>1. МКЭ, стержневой элемент, ферма </h1>
# 
# $$ \int \delta K - (\delta U - \delta W) \; dt = 0 $$
# 
# $$ M \ddot{u} + K u = f $$
# $$ K^{e} = \dfrac{EA}{2a} \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}, \quad M^{e} = \dfrac{\rho A}{3} a \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}, \quad f^{e} = a p \begin{bmatrix} 1 \\ 1 \end{bmatrix}. $$
# 

from matplotlib import  pyplot as plt
import numpy as np
import scipy.linalg as LA

E = 30e6; A = 1.; EA = E*A; L = 90.; p = 50.

numberElements = 10
nodeCoordinates = np.linspace(0, L, numberElements+1, endpoint=True)
numberNodes = len(nodeCoordinates)

#connectivity
elementNodes = np.zeros([numberElements, 2], dtype = int)
for i in range(numberElements): 
    elementNodes[i, 0] =i
    elementNodes[i, 1] =i+1

displacements = np.zeros(numberNodes)
force = np.zeros(numberNodes)
stiffness = np.zeros([numberNodes, numberNodes])


def shapeFunctionL2(xi):
    my_shape = np.array([[(1-xi)/2,(1+xi)/2 ]])
    my_Xderivatives = np.array([[ -1./2 , 1./2 ]])
    return my_shape.T, my_Xderivatives
    
def solve(GDof, prescribedDof, stiffness, force):
    activeDof = np.setdiff1d(np.arange(GDof), prescribedDof)
    # Exclude prescribed DOFs, solve the system
    A = stiffness[activeDof,:][:, activeDof]
    F = force[activeDof]
    disp = LA.solve(A,F)
    return disp


# Construct Stifness matrix
for i in range(numberElements):
    elementDof = elementNodes[i,:]
    nn = len(elementDof)
    length_element = nodeCoordinates[elementDof[1]] - nodeCoordinates[elementDof[0]]
    detJacobian = length_element/2
    invJacobian = 1./detJacobian

    # central Gauss point (xi=0, weight W=2)
    shape, naturalDerivatives = shapeFunctionL2(0.0)
    Xderivatives = naturalDerivatives*invJacobian

    #
    B = Xderivatives.copy()
    BtB = 2*(B.T@B) * detJacobian * EA
    stiffness[elementDof[0], elementDof] += BtB[0]
    stiffness[elementDof[1], elementDof] += BtB[1]
    force[elementDof[0]] += 2*p*detJacobian*shape[0]
    force[elementDof[1]] += 2*p*detJacobian*shape[1]
    
# Prescribed DOFs
allDof = np.arange(numberNodes)
prescribedDof = [0, numberNodes-1]
activeDof = np.setdiff1d(allDof, prescribedDof)

GDof = numberNodes
displacements[activeDof] = solve(GDof, prescribedDof, stiffness, force)
sigma = np.zeros(numberElements)

def exact_solution(x):
    u = p*L*x / (2*E*A)*(1 - x / L)
    s_x =  p*L/A*(0.5 - x/L)
    return u, s_x

print(displacements)
print(exact_solution(nodeCoordinates)[0])


x_ = np.linspace(0, L, 100)
ex_ans = exact_solution(x_)
plt.plot(x_, ex_ans[0], label = "Ex")
plt.plot(nodeCoordinates, displacements, label ="Num")
plt.scatter(nodeCoordinates, displacements)
plt.legend()
plt.show()
