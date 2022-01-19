import numpy as np

def KitaevMatrixElements(j1,j2,j3, state, periodic = False):

    """
    Computes the matrix element of the 2 leg-ladder Kitaev model mapped into a 1d system for a given state
    Uses the following mapping:
    1-j1-2-j2-5-j1-6
    |    |    |    |
    0-j2-3-j1-4-j2-7
    Vertical lines == j3 interaction
    -----------------------------------------------------------------------------------
    Parameters:
    j1, j2, j3: floats, Kitaev parameters
    state:      np.ndarrray of dtype=int and shape (N)
                spin-state, integer encoded (using 0 for down spin and 1 for up spin)
                A sample of spins can be fed here.
    periodic: False = open boundary conditions and True = periodic boundary conditions
    -----------------------------------------------------------------------------------            
    Returns: 2-tuple of type (np.ndarray,np.ndarray)
             sigmas:         np.ndarray of dtype=int and shape (?,N)
                             the states for which there exist non-zero matrix elements for given sigmap
             matrixelements: np.ndarray of dtype=float and shape (?)
                             the non-zero matrix elements
    """

    sigmas=[]
    matrixelements=[]
    N=len(state)

    #the diagonal part is the sum of all Sz-Sz interactions which are always
    #between a even-indexed spin and the next one
    if periodic:
        limit = N
    else:
        limit = N-1
    
    N_half = int(N/2)
    diag = 0 
    for site in range(N_half):
        if state[2 * site]!=state[2 * site+1]: #if the two neighouring spins are opposite
            diag-=0.25*j3 #add a negative energy contribution
        else:
            diag+=0.25*j3

    matrixelements.append(diag) #add the diagonal part to the matrix elements
    sigmas.append(np.copy(state))

    #off-diagonal part:
    #j1 interactions
    for site in range(1, N_half):
        if j1 != 0.0:
            #Flip spin 2j and 2j-1
            sig=np.copy(state)
            sig[2 * site]  = 1 - state[2 * site]
            sig[2 * site-1]= 1 - state[2 * site-1]
            sigmas.append(sig) 
            matrixelements.append(j1 * 0.25)
    
    for site in range(N_half-1):
        if j2 != 0.0:
            #Flip spin 2j and 2j+3
            #the sign is "-" if equal spins, else "+"
            sig=np.copy(state)
            sig[2 * site]  = 1 - state[2 * site]
            sig[2 * site+3]= 1 - state[2 * site+3]
            sigmas.append(sig)
            sign = -1 if state[2 * site] == state[2 * site+3] else +1
            matrixelements.append(sign * j2 * 0.25)

    return np.array(sigmas),np.array(matrixelements)

def KitaevRungMatrixElements(j1,j2,j3, state, periodic = False):

    """
    Computes the matrix element of the 2 leg-ladder Kitaev model mapped into a 1d system for a given state
    The mapping is made by means of Rungs (=2 vertical spins paired)
    Spins are 1/2
    -----------------------------------------------------------------------------------
    Parameters:
    j1, j2, j3: floats, Kitaev parameters
    state:      np.ndarrray of dtype=int and shape (N)
                spin-state, integer encoded (using the following map:
                                            0 = dd      2 = ud
                                            1 = du      3 = uu
                                        )
                A sample of spins can be fed here
    periodic: False = open boundary conditions and True = periodic boundary conditions
    -----------------------------------------------------------------------------------            
    Returns: 2-tuple of type (np.ndarray,np.ndarray)
             sigmas:         np.ndarray of dtype=int and shape (?,N)
                             the states for which there exist non-zero matrix elements for given sigmap
             matrixelements: np.ndarray of dtype=float and shape (?)
                             the non-zero matrix elements
    """

    sigmas=[]
    matrixelements=[]
    N=len(state)

    if periodic:
        limit = N
    else:
        limit = N-1
    diag = 0 
    for site in range(N):
        if state[site]== 1 or state[site] == 2: #if the two neighouring spins are opposite
            diag-=0.25*j3 #add a negative energy contribution
        else:
            diag+=0.25*j3

    matrixelements.append(diag) #add the diagonal part to the matrix elements
    sigmas.append(np.copy(state))


    #off-diagonal part
    for site in range(N - 1):
        if (site%2==0):

            #Flip spin 2j and 2j-1
            sig=np.copy(state)
            sig[site]  = (2 + state[site]) % 4
            sig[site +1]  = (2 + state[site +1]) % 4
            sigmas.append(sig) 
            matrixelements.append(j1 * 0.25)

            sig=np.copy(state)
            sig[site] += 1 - 2 * (sig[site] % 2)
            sig[site+1] += 1 - 2 * (sig[site+1] % 2)
            sigmas.append(sig) 
            sign = 1 - 2 * ( (sig[site] + sig[site+1])%2 == 0 )
            matrixelements.append(sign * j2 * 0.25)
    
        else:
            sig=np.copy(state)
            sig[site]  = (2 + state[site]) % 4
            sig[site +1]  = (2 + state[site +1]) % 4
            sigmas.append(sig) 
            sign = 1 - 2 * ( (sig[site] == sig[site+1]) |  ((sig[site] + sig[site+1])%4 == 1) )
            matrixelements.append(sign * j2 * 0.25)

            sig=np.copy(state)
            sig[site] += 1 - 2 * (sig[site] % 2)
            sig[site+1] += 1 - 2 * (sig[site+1] % 2)
            sigmas.append(sig) 
            matrixelements.append(j1 * 0.25)
        

    return np.array(sigmas),np.array(matrixelements)

def ED_Kitaev(N, j1 = 1.0, j2 =1.0, j3 = 1.0, periodic=False):
    """
    Returns a tuple (eta,U)
      eta = a list of energy eigenvalues.
      U = a list of energy eigenvectors
    """
    basis = []

    #Generate a basis
    for i in range(2**N):
        basis_temp = np.zeros((N))
        a = np.array([int(d) for d in bin(i)[2:]])
        l = len(a)
        basis_temp[N-l:] = a

        basis.append(basis_temp)

    basis = np.array(basis)


    H=np.zeros((basis.shape[0],basis.shape[0])) #prepare the hamiltonian
    for n in range(basis.shape[0]):
        sigmas,elements= KitaevMatrixElements(j1, j2, j3,np.reshape(basis[n,:],(N)),periodic)
        for m in range(sigmas.shape[0]):
            for b in range(basis.shape[0]):
                if np.all(basis[b,:]==sigmas[m,:]):
                    H[b,n]=elements[m]
                    #print("Setting Elements")
    print(basis[5], H[0,5])
    eta,U=np.linalg.eigh(H) #diagonalize
    return eta,U


def ED_Kitaev_Rung(N, j1 = 1.0, j2 =1.0, j3 = 1.0, periodic=False):
    """
    Returns a tuple (eta,U)
      eta = a list of energy eigenvalues.
      U = a list of energy eigenvectors
    """
    basis = []

    #Generate a basis
    for i in range(4**N):
        basis_temp = np.zeros((N))
        a = np.array([int(d) for d in np.base_repr(i, base=4)])
        l = len(a)
        basis_temp[N-l:] = a

        basis.append(basis_temp)

    basis = np.array(basis)


    H=np.zeros((basis.shape[0],basis.shape[0])) #prepare the hamiltonian
    for n in range(basis.shape[0]):
        sigmas,elements= KitaevRungMatrixElements(j1, j2, j3,np.reshape(basis[n,:],(N)),periodic)
        for m in range(sigmas.shape[0]):
            for b in range(basis.shape[0]):
                if np.all(basis[b,:]==sigmas[m,:]):
                    H[b,n]=elements[m]
                    break
    print(basis[6], H[0,6])
    eta,U=np.linalg.eigh(H) #diagonalize
    return eta,U