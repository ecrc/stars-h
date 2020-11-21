import pandas as pd
import numpy as np
import math
import os.path
from os import rename
from scipy.stats import ortho_group
import pyHilbertSort as m

a    = 1.7
rad  = 0.18/2


def E(x):
    return math.floor(x + 0.5)

NbPerz = E(3*(a)/(2*rad*math.sqrt(6)))
NbPery = E((a)/(rad*math.sqrt(3)))
NbPerx = E((a)/(2*rad))

MaxnbVirus = NbPerx*NbPery*NbPerz
MaxDensity = (4*math.pi*math.pow(rad,3)*MaxnbVirus/3)/math.pow(a,3)

def E(x):
    return math.floor(x + 0.5)

path = input("Please enter the mesh file path : ")
while(not os.path.isfile(path)):
    print("File not accessible")
    path = input("Please enter the mesh file path : ")
M = pd.read_csv(path ,sep = ',',header = None)

print("For this cube length, the maximum possible density is {d} with {nb} viruses ".format(nb = MaxnbVirus, d = MaxDensity))

fixedparam = input("Do you prefer to fixe the density(1) or the number of viruses(2) ?  (1/2) : ")
while(not fixedparam.isdigit()):
    print("Non valid entry")
    fixedparam = input("Do you prefer to fixe the density(1) or the number of viruses(2) ?  (1/2) : ")
fixedparam = int(fixedparam)

if fixedparam == 1 :
    Density = input("Please enter the virus density : ")
    while(float(Density) > MaxDensity):
        print("Non valid entry")
        Density = input("Please enter the virus density : ")
    nbVirus = int(float(Density)*math.pow(a,3)/(4*math.pi*math.pow(rad,3)/3))
    print("The total number of viruses = {w}".format(w = nbVirus))
else :
    nbVirus = input("Please enter the total number of viruses : ")
    while(float(nbVirus) > MaxnbVirus):
        print("Non valid entry")
        nbVirus = input("Please enter the total number of viruses : ")
    nbVirus = int(nbVirus)
    print("The density = {w}".format(w = (4*math.pi*math.pow(rad,3)*nbVirus/3)/math.pow(a,3)))
    dens=(4*math.pi*math.pow(rad,3)*nbVirus/3)/math.pow(a,3)
    print("dens", dens)
    Density=dens
ordering = input("Do you prefer to include Hilbert ordering (y/n) : ")
while(ordering != 'y') and (ordering != 'n'):
    print("Non valid entry")
    ordering = input("Do you prefer to include Hilbert ordering (y/n) : ")

cube = []
for k in range(0 ,NbPerz):
    for j in range(0 ,NbPery):
        for i in range(0 ,NbPerx):
            x    = (2*i + (j+k)%2)*rad
            y    = (math.sqrt(3)*(j+(k%2)/3))*rad
            z    = (2*math.sqrt(6)/3)*k*rad
            cube.append([x, y, z])
pos = np.asarray(cube)

newPath = "data/VirusPopulation.txt"

virus = np.unique(np.random.randint(0, len(cube), size=nbVirus), axis=0)
while (len(virus) < nbVirus):
    virus = np.append(virus,np.random.randint(0, len(cube), size = nbVirus - len(virus)), axis = 0)
    virus = np.unique(virus, axis = 0)
VirusPos = pos[virus]

T       = ortho_group.rvs(3)
np.savetxt(newPath, (np.dot(T, (M.values).T ).T + VirusPos[0] ), delimiter=",", fmt='%1.4e')

f = open(newPath,'ab')
for i in range(1,nbVirus):
    T = ortho_group.rvs(3)
    np.savetxt(f, (np.dot(T, (M.values).T ).T + VirusPos[i] ), delimiter=",", fmt='%1.4e')
f.close()
rename(newPath,"data/VirusPopulation"+str(nbVirus*M.shape[0])+"_"+str(Density)+".txt")

if(ordering == "y"):
    ordPath = "data/VirusPopulation"+str(nbVirus*M.shape[0])+"_"+str(Density)+".txt"
    pts = pd.read_csv(ordPath ,sep = ',',header = None)
    ptsSort,idxSort = m.hilbertSort(3, pts.values)
    np.savetxt(ordPath, ptsSort, delimiter=",", fmt='%1.4e')
