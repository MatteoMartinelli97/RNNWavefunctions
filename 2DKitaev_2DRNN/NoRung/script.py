import numpy as np
import ExactDiagonalization_Kitaev2D as ed



Nx = 3
Ny = 4

nrgs = []
d = []

j1s = np.linspace(0.0, 2.0, num=101, endpoint=True)
j2s = []
j1, j2, j3 = (0.0, 2.0, 1.0)
j1s = [1.0]
for j1 in j1s:
    
    j2 = -j1 + 2.0
    j2s.append(j2)
    print("j1 = ", j1, "j2 = ", j2)
    e, vec = ed.ED_Kitaev2D(Nx=Nx, Ny=Ny, j1 = j1, j2 =j2, j3 = j3)
    print(e[0])
    nrgs.append(e[0])
    #d.append(dx)


#np.save("j1s.npy", j1s)
#np.save("j2s.npy", j2s)
#np.save("{}x{}_nrg.npy".format(Nx,Ny), nrgs)
#np.save("{}x{}_delta.npy".format(Nx, Ny), d)
