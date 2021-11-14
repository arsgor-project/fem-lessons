# from MATLAB codes for Finite Element Analysis
# A.J.M. Ferreira, N. Fantuzzi 2019
# Trusses in 2D

import numpy as np
import math 
import scipy.linalg as LA
import time
from matplotlib import  pyplot as plt

def formStiffness2Dtruss(GDof_, numberElements_, elementNodes_, nodeCoordinates_, xx_, yy_, EA_):
    stiffness= np.zeros([GDof_, GDof_])
    
    for e  in range(numberElements_):
        indice = elementNodes_[e, :]
        elementDof = [indice[0]*2 , indice[0]*2+1, indice[1]*2, indice[1]*2 + 1] 
        xa = xx_[indice[1]] - xx_[indice[0]]
        ya = yy_[indice[1]] - yy_[indice[0]]
        length_element = math.sqrt(xa*xa+ya*ya);
        C = xa/length_element
        S = ya/length_element
        k1 = EA_/length_element * np.array( [[C*C, C*S, -C*C, -C*S],
                                            [C*S, S*S, -C*S, -S*S],
                                            [-C*C, -C*S, C*C, C*S],
                                            [-C*S, -S*S, C*S, S*S]])
        stiffness[elementDof[0], elementDof] += k1[0]
        stiffness[elementDof[1], elementDof] += k1[1]
        stiffness[elementDof[2], elementDof] += k1[2]
        stiffness[elementDof[3], elementDof] += k1[3]
    
    return stiffness
        
def solve(GDof_, prescribedDof_, stiffness_, force_):
    activeDof = np.setdiff1d(np.arange(GDof_), prescribedDof_)
    # Exclude prescribed DOFs, solve the system
    A = stiffness_[activeDof,:][:, activeDof]
    F = force_[activeDof]
    disp = np.zeros(GDof) 
    disp[activeDof] = LA.solve(A,F)
    return disp

def computeReactions(stiffness_, displacements_):
    return stiffness_@displacements

def stresses2Dtruss(numberElements_, elementNodes_, xx_, yy_, displacements_, E_):
    sigma_ = np.zeros(numberElements_)
    for e  in range(numberElements_):
        indice = elementNodes_[e, :]
        elementDof = [indice[0]*2 , indice[0]*2+1, indice[1]*2, indice[1]*2 + 1] 
        xa = xx_[indice[1]] - xx_[indice[0]]
        ya = yy_[indice[1]] - yy_[indice[0]]
        length_element = math.sqrt(xa*xa+ya*ya);
        C = xa/length_element
        S = ya/length_element
        sigma_[e] = E_/length_element*np.array([-C,-S,C,S])@displacements_[elementDof]
    return sigma_

def plot2Dtruss(numberElements_, elementNodes_, xx_, yy_, displacements_):
    fig, ax = plt.subplots()
    ax.scatter(xx_, yy_, label = 'undeformed')
    
    for e in range(numberElements_):
        ax.plot(xx_[elementNodes_[e]], yy_[elementNodes_[e]], color = 'black')
    
    scale = 1e2
    new_xx_ = xx_ + scale*displacements_[::2]
    new_yy_ = yy_ + scale*displacements_[1::2]
    ax.scatter(new_xx_, new_yy_, label = 'deformed')
    
    for e in range(numberElements_):
        ax.plot(new_xx_[elementNodes_[e]], new_yy_[elementNodes_[e]], '--' ,color = 'gray')

    fig.legend()
    plt.show()

E = 30e6; A = 2; EA = E*A

elementNodes = np.array([[0, 1],[0, 2], [1, 2], [1,3], [0,3], [2,3], [2,5], [3,4], [3,5], [2,4], [4,5]])
nodeCoordinates = np.array([ [0,0], [0,3000], [3000,0], [3000,3000], [6000, 0], [6000, 3000]])
numberElements = len(elementNodes)
numberNodes = len(nodeCoordinates)

xx = nodeCoordinates[:, 0]
yy = nodeCoordinates[:, 1]

# total number of degrees of freedom
GDof = 2*numberNodes 
displacements = np.zeros(GDof)
force = np.zeros(GDof)
force[3] = -5e4
force[7] = -1e5
force[11] = -5e4
prescribedDof = np.array([0,1,9])


start = time.time()
stiffness = formStiffness2Dtruss(GDof,numberElements, elementNodes,nodeCoordinates, xx, yy, EA)
displacements = solve(GDof, prescribedDof, stiffness,force)
reactions = computeReactions(stiffness, displacements)
sigma = stresses2Dtruss(numberElements, elementNodes, xx, yy, displacements, E)

end = time.time()

print("Finished. Time in seconds = ", end - start)
print("Reactions:", reactions)
print("Stresses:", sigma)
plot2Dtruss(numberElements, elementNodes, xx, yy, displacements)

