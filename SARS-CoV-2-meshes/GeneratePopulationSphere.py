import pandas as pd
import numpy as np
import math
import os.path
from os import rename
from scipy.stats import ortho_group
import pyHilbertSort as m

path = input("Please enter the mesh file path : ")
while(not os.path.isfile(path)):
    print("File not accessible")
    path = input("Please enter the mesh file path : ")
M = pd.read_csv(path ,sep = ',',header = None)

nbVirus = input("Please enter the number of viruses : ")
while(not nbVirus.isdigit()):
    print("Non valid entrie")
    nbVirus = input("Please enter the number of viruses : ")
nbVirus = int(nbVirus)

Density = float(input("Please enter the virus density : "))

ordering = input("Do you prefer to include Hilbert ordering (y/n) : ")
while(ordering != 'y') and (ordering != 'n'):
    print("Non valid entrie")
    ordering = input("Do you prefer to include Hilbert ordering (y/n) : ")

rad  = 0.18/2 #rad of cube
#rad  = 0.14/2
a    = math.pow(((4*math.pi*math.pow(rad,3)*nbVirus/3) / Density ),1/3)
cube = []

def E(x):
    return math.floor(x + 0.5)

NbPerz   = E(3*(a)/(2*rad*math.sqrt(6)))
NbPery   = E((a)/(rad*math.sqrt(3)))
NbPerx   = E((a)/(2*rad))

for k in range(0 ,NbPerz):
    for j in range(0 ,NbPery):
        for i in range(0 ,NbPerx):
            x    = (2*i + (j+k)%2)*rad
            y    = (math.sqrt(3)*(j+(k%2)/3))*rad
            z    = (2*math.sqrt(6)/3)*k*rad
            cube.append([x, y, z])
pos = np.asarray(cube)

newPath = "case1sphere/Case1sphereVirusPopulation.txt"

if NbPerz*NbPery*NbPerx < nbVirus :
    VirusPos    = pos
    nbVirus     = NbPerz*NbPery*NbPerx
    TrueDensity = (4*math.pi*math.pow(rad,3)*nbVirus/3)/math.pow(a,3)
    VirusPos    = pos
    print("For this set of parameters, the maximum possible density is {d} with {nb} viruses ".format(nb = nbVirus, d = TrueDensity))
else :
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
rename(newPath,"case1sphere/Case1sphereVirusPopulation"+str(nbVirus*M.shape[0])+"_"+str(Density)+".txt")

if(ordering == "y"):
    ordPath = "case1sphere/Case1sphereVirusPopulation"+str(nbVirus*M.shape[0])+"_"+str(Density)+".txt"
    pts = pd.read_csv(ordPath ,sep = ',',header = None)
    ptsSort,idxSort = m.hilbertSort(3, pts.values)
    np.savetxt(ordPath, ptsSort, delimiter=",", fmt='%1.4e')
