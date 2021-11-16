import numpy as np
from matplotlib import  pyplot as plt

#E; modulus of elasticity
#I: second moment of area
#L: length of bar
E = 1; I = 1; EI=E*I

numberElements = 2
nodeCoordinates = np.linspace(0,1,numberElements+1)
L = np.max(nodeCoordinates)
numberNodes = len(nodeCoordinates)
xx = nodeCoordinates[:]
elementNodes = np.zeros([numberElements,2])
for i in range(numberElements):
    elementNodes[i,0]=i
    elementNodes[i,1]=i+1

#distributed load
P = -1

def formStiffnessBernoulliBeam(GDof, numberElements, elementNodes, numberNodes, xx, EI, P):
    stiffness= np.zeros([GDof, GDof])
    force = np.zeros(GDof)

    for e  in range(numberElements):
        indice = elementNodes[e, :]
        elementDof = [ int(indice[0])*2 , int(indice[0])*2+1, int(indice[1])*2, int(indice[1])*2 + 1] 
        length_element = abs(xx[int(indice[1])] - xx[int(indice[0])])
        a = length_element/2
        k1 = EI/(2*a**3) * np.array( [[3, 3*a, -3, 3*a],
                                            [3*a, 4*a**2, -3*a, 2*a**2],
                                            [-3, -3*a, 3,  -3*a],
                                            [3*a, 2*a**2, -3*a, 4*a**2]])
        stiffness[elementDof[0], elementDof] += k1[0]
        stiffness[elementDof[1], elementDof] += k1[1]
        stiffness[elementDof[2], elementDof] += k1[2]
        stiffness[elementDof[3], elementDof] += k1[3]

        f1 = a*P/3*np.array([3,a,3,-a])
        force[elementDof] +=f1
    return stiffness, force

def solve(GDof_, prescribedDof_, stiffness_, force_):
    activeDof = np.setdiff1d(np.arange(GDof_), prescribedDof_)
    # Exclude prescribed DOFs, solve the system
    A = stiffness_[activeDof,:][:, activeDof]
    F = force_[activeDof]
    disp = np.zeros(GDof) 
    disp[activeDof] = np.linalg.solve(A,F)
    return disp

def plotBeam(xx_, W, R):
    fig, ax = plt.subplots()
    ax.plot(xx_, W, color = 'black')
    fig.legend()
    plt.show()

#GDof: global number of degrees of freedom
GDof = 2*numberNodes
#stiffess matrix and force vector
stiffness, force = formStiffnessBernoulliBeam(GDof,numberElements,elementNodes,numberNodes,xx,EI,P)

#boundary conditions and solution
fixedNode =np.array([0,1, GDof-2, GDof-1])
prescribedDof = fixedNode

#solution
displacements = solve(GDof,prescribedDof,stiffness,force)
W = displacements[::2]
R = displacements[1::2]
print(W)
exact = 1/384*P*L**4/EI
print(exact)
plotBeam(xx, W, R)