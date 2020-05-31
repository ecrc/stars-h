import pandas as pd
import numpy as np
import os.path
from os import rename


path = input("Please enter the mesh file path : ")
while(not os.path.isfile(path)):
    print("File not accessible")
    path = input("Please enter the mesh file path : ")

outputfile = input("Please enter the path to the output file: ")
while(not isinstance(outputfile, str)):
    print("Output File not accessible")
    path = input("Please enter the path to the output file: ")
    
M = pd.read_csv(path ,sep = ',',header = None)

nb = input("Please enter the number of viruses : ")
while(not nb.isdigit()):
    print("Non valid entrie")
    nb = input("Please enter the number of viruses : ")
    
nb = int(nb)
Rmax = int(nb/2)
Rmin = -int(nb/2)
stp  = 0.25

# Generate population of viruses that are uniformally dist with dist 0.25 from the center of virus

pos = np.unique(np.random.randint(Rmin, Rmax, size=(nb,3))*stp, axis=0)
while (len(pos) < nb):
    pos = np.vstack((pos,np.random.randint(Rmin, Rmax, size=(1,3))*stp))
    pos = np.unique(pos, axis=0)
    
    
newPath = "data/VirusPopulation.txt"
np.savetxt(newPath, (M + pos[0]).values, delimiter=",", fmt='%1.4e')

f = open(newPath,'ab')
for i in range(1,nb):
    np.savetxt(f, (M + pos[i]).values, delimiter=",", fmt='%1.4e')
f.close()
rename(newPath, outputfile)
