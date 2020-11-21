# @version 0.3.0

import pandas as pd
import numpy as np
import os.path
from os import rename
from scipy.stats import ortho_group
import pyHilbertSort as m

path = input("Please enter the mesh file path : ")
while(not os.path.isfile(path)):
    print("File not accessible")
    path = input("Please enter the mesh file path : ")

M = pd.read_csv(path ,sep = ',',header = None)

outputfile = input("Please enter the path to the output file: ")
while(not isinstance(outputfile, str)):
    print("Output File not accessible")
    path = input("Please enter the path to the output file: ")

globnb = input("Please enter the total number of viruses : ")
while(not globnb.isdigit()):
    print("Non valid entrie")
    globnb = input("Please enter the total number of viruses : ")

nb = input("Please enter the number of viruses per sub-population : ")
while(not nb.isdigit()) and (nb > globnb):
    print("Non valid entrie")
    nb = input("Please enter the number of viruses per sub-population : ")

ordering = input("Do you prefer to include Hilbert ordering (y/n) : ")
while(ordering != 'y') and (ordering != 'n'):
    print("Non valid entrie")
    ordering = input("Do you prefer to include Hilbert ordering (y/n) : ")


#Local population
nb = int(nb)
Rmax = nb
Rmin = 0
stp  = 0.25

#Global population
globnb = int(globnb)
popnb  = int(np.ceil(globnb/nb))
globRmax = np.ceil((popnb+1)**(1/3))
globRmin = 0
globstp  = (Rmax-Rmin)*stp

center = np.unique(np.random.randint(globRmin, globRmax, size=(popnb,3))*globstp, axis=0)
while (len(center) < popnb):
    center = np.vstack((center,np.random.randint(globRmin, globRmax, size=(1,3))*globstp))
    center = np.unique(center, axis=0)

newPath = "singleviursdata/VirusPopulation.txt"
countt=0
for c in center :
    T = ortho_group.rvs(3)

    pos = np.unique(np.random.randint(Rmin, Rmax, size=(nb,3))*stp, axis=0)
    while (len(pos) < nb):
        pos = np.vstack((pos,np.random.randint(Rmin, Rmax, size=(1,3))*stp))
        pos = np.unique(pos, axis=0)

    np.savetxt(newPath, (np.dot(T, (M.values).T ).T + pos[0] + c ), delimiter=",", fmt='%1.4e')

    f = open(newPath,'ab')
    for i in range(1,nb):
        T = ortho_group.rvs(3)
        np.savetxt(f, (np.dot(T, (M.values).T ).T + pos[i] + c), delimiter=",", fmt='%1.4e')
    f.close()
    rename(newPath, outputfile+str(countt)+".txt")
#    rename(newPath,"data/VirusPopulation"+str(nb*M.shape[0])+str(c)+".txt")

    if(ordering == "y"):
        #ordPath = "data/VirusPopulation"+str(nb*M.shape[0])+str(c)+".txt"
        ordPath=outputfile+str(countt)+".txt"
        pts = pd.read_csv(ordPath ,sep = ',',header = None)
        ptsSort,idxSort = m.hilbertSort(3, pts.values)
        np.savetxt(ordPath, ptsSort, delimiter=",", fmt='%1.4e')
    countt=countt+1
