import numpy as np
import scipy.linalg as splinalg

def Kitaev2DMatrixElements(j1,j2,j3, state):

    """
    Computes the matrix element of the 2D Kitaev model for a given state
    Uses the following mapping:
    0-j1-1-j2-2-j1-3
    |         |    
    4-j2-5-j1-6-j2-7
         |         |
    8-x--9-y-10-x-11
    |         |     
    12-x-13-y-14-x-15

    Vertical lines == j3 interaction
    -----------------------------------------------------------------------------------
    Parameters:
    j1, j2, j3: floats, Kitaev parameters
    state:      np.ndarrray of dtype=int and shape (Ny, Nx)
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
    Nx = state.shape[1]
    Ny = state.shape[0]

    Nx_half = int(Nx *0.5)

    #the diagonal part is the sum of all Sz-Sz interactions which are on even spins for j even and 
    #odd spins for j odd
    
    diag = 0 
    for i in range(Ny):
        #loop over rows
        if i%2==0:
            #even row, interction on odd spins with previous row
            val = 1 - 2 * ( (state[i][1::2] + state[i-1][1::2])%2 ) #1 if both 0 or 1 spins, -1 otherwise
            diag += j3 * np.sum(val)
        else:
            #odd row, interction on even spins with previous row
            val = 1 - 2 * ( (state[i][0::2] + state[i-1][0::2])%2 )  #1 if both 0 or 1 spins, -1 otherwise
            diag += j3 * np.sum(val)

    matrixelements.append(diag) #add the diagonal part to the matrix elements
    sigmas.append(np.copy(state))


    #off-diagonal part:
    for i in range(Ny):
        #loop over rows
        if i%2==0:
            #on even rows interaction is xx--yy--xx...
            for j in range(Nx_half):
                #loop over columns
                sig=np.copy(state)
                #xx on spin 2j and 2j+1
                sig[i][2*j]  = 1 - state[i][2*j]
                sig[i][2*j+1]= 1 - state[i][2*j+1]
                sigmas.append(sig)
                matrixelements.append(j1)
            
                sig=np.copy(state)
                #yy on spin 2j+1 and 2j+2
                sig[i][2*j+1]  = 1 - state[i][2*j+1]
                sig[i][2*(j+1)]= 1 - state[i][2*(j+1)]
                sign = -1 if state[i][2*j+1] == state[i][2*(j+1)] else +1
                sigmas.append(sig)
                matrixelements.append(sign*j2)
        else:
            #on odd rows interaction is yy--xx--yy...
            for j in range(Nx_half):
                #loop over columns
                sig=np.copy(state)
                #yy on spin 2j and 2j+1
                sig[i][2*j]  = 1 - state[i][2*j]
                sig[i][2*j+1]= 1 - state[i][2*j+1]
                sign = -1 if state[i][2*j] == state[i][2*j+1] else +1
                sigmas.append(sig)
                matrixelements.append(sign*j2)
            
                sig=np.copy(state)
                #xx on spin 2j+1 and 2j+2
                sig[i][2*j+1]  = 1 - state[i][2*j+1]
                sig[i][2*(j+1)]= 1 - state[i][2*(j+1)]
                sigmas.append(sig)
                matrixelements.append(j1)
               

    return np.array(sigmas),np.array(matrixelements)

def Kitaev2DRungMatrixElements(j1,j2,j3, state):

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
    
    Nx = state.shape[1]
    Ny = state.shape[0]
    
    diag = 0 
    for j in range(0, Nx, 2):
        #diagonal elements even for intra-rung
        val = np.copy(state[:,j])
        val[state[:,j]==1] = -1 #Opposite spin
        val[state[:,j]==2] = -1 #Opposite spin
        val[state[:,j]==3] = +1 #Same spin
        val[state[:,j]==0] = +1 #Same spin
        #sum over y-axis = sum over all the rows
        diag += np.sum(val*j3)

    for i in range(Ny):
        #diagonal elements odd for inter-rung
        #first and last row are linked by PBC 
        valuesA = np.copy(state[i-1, 1::2])

        #Map to actual spins and compute the interaction on z-axis
        #Upper row, take the lower spin
        valuesA[state[i-1, 1::2]==0] = -1
        valuesA[state[i-1, 1::2]==2] = -1
        valuesA[state[i-1, 1::2]==1] = +1
        valuesA[state[i-1, 1::2]==3] = +1

        valuesB = np.copy(state[i, 1::2])
        #Lower row, take the upper spin
        valuesB[state[i, 1::2]==0] = -1
        valuesB[state[i, 1::2]==1] = -1
        valuesB[state[i, 1::2]==2] = +1
        valuesB[state[i, 1::2]==3] = +1

        diag += np.sum(j3 * valuesA * valuesB)

    matrixelements.append(diag) #add the diagonal part to the matrix elements
    sigmas.append(np.copy(state))

    print(diag)
    #off-diagonal part
    for i in range(Ny):
        for j in range(Nx-1):
            #off-diag interactions are always on the same row i
            if (j%2==0):
                #Flip rung j and j+1
                #Upper - xx
                sig=np.copy(state)
                sig[i, j]  = (2 + state[i, j]) % 4
                sig[i, j+1]  = (2 + state[i, j +1]) % 4
                sigmas.append(sig) 
                matrixelements.append(j1)

                # Lower - yy 
                sig=np.copy(state)
                sig[i, j] += 1 - 2 * (sig[i, j] % 2)
                sig[i, j+1] += 1 - 2 * (sig[i, j+1] % 2)
                sigmas.append(sig) 
                sign = 1 - 2 * ( (sig[i, j] + sig[i, j+1])%2 == 0 )
                matrixelements.append(sign * j2)

            else:
                #Upper - yy
                sig=np.copy(state)
                sig[i, j]  = (2 + state[i, j]) % 4
                sig[i, j +1]  = (2 + state[i, j +1]) % 4
                sigmas.append(sig) 
                sign = 1 - 2 * ( (sig[i, j] == sig[i, j+1]) |  ((sig[i, j] + sig[i, j+1])%4 == 1) )
                print(sign)
                matrixelements.append(sign * j2)

                # Lower - xx
                sig=np.copy(state)
                sig[i, j] += 1 - 2 * (sig[i, j] % 2)
                sig[i, j+1] += 1 - 2 * (sig[i, j+1] % 2)
                sigmas.append(sig) 
                matrixelements.append(j1)
        

    return np.array(sigmas),np.array(matrixelements)

def ED_Kitaev2D(Nx, Ny, j1 = 1.0, j2 =1.0, j3 = 1.0):
    """
    Returns a tuple (eta,U)
      eta = a list of energy eigenvalues.
      U = a list of energy eigenvectors
    """
    basis = []
    N = Nx*Ny
    #Generate a basis
    for i in range(2**N):
        basis_temp = np.zeros((N))
        a = np.array([int(d) for d in bin(i)[2:]])
        l = len(a)
        basis_temp[N-l:] = a

        basis.append(basis_temp)


    basis = np.array(basis) #(2^N, N)
    H=np.zeros((basis.shape[0],basis.shape[0]), dtype=np.complex64) #prepare the hamiltonian
    for n in range(basis.shape[0]):
        sigmas,elements= Kitaev2DMatrixElements(j1, j2, j3,np.reshape(basis[n,:],(Ny, Nx)))
        for m in range(sigmas.shape[0]):
            for b in range(basis.shape[0]):
                if np.all(basis[b,:]==np.reshape(sigmas[m,:], N)):
                    H[b,n]=elements[m]
                    break
    #print(H)
    eta,U=np.linalg.eigh(H) #diagonalize
    return eta,U


def ED_Kitaev2D_Rung(Nx, Ny, j1 = 1.0, j2 =1.0, j3 = 1.0):
    """
    Returns a tuple (eta,U)
      eta = a list of energy eigenvalues.
      U = a list of energy eigenvectors
    """
    basis = []
    N = Nx*Ny

    #Generate a basis
    for i in range(4**N):
        basis_temp = np.zeros((N))
        a = np.array([int(d) for d in np.base_repr(i, base=4)])
        l = len(a)
        basis_temp[N-l:] = a

        basis.append(basis_temp)

    basis = np.array(basis)


    H=np.zeros((basis.shape[0],basis.shape[0]), dtype=np.complex64) #prepare the hamiltonian
    for n in range(basis.shape[0]):
        sigmas,elements= Kitaev2DRungMatrixElements(j1, j2, j3,np.reshape(basis[n,:],(Ny, Nx)))
        for m in range(sigmas.shape[0]):
            for b in range(basis.shape[0]):
               if np.all(basis[b,:]==np.reshape(sigmas[m], N)):
                    H[b,n]=elements[m]
                    break
    eta,U=np.linalg.eigh(H) #diagonalize
    return eta,U


def DeltaX_MatrixElement( state, periodic = False):
    """
    Given "state" returns the only non-zero state when the DeltaX operator is applied
    Uses the following mapping:
    1-j1-2-j2-5-j1-6
    |    |    |    |
    0-j2-3-j1-4-j2-7
    DeltaX is defined as PI_i=1^2j Sx_i
    ---------------------------------------------------------------------------------------
    Returns: 2-tuple of type (np.ndarray,np.ndarray)
         sigmas:         np.ndarray of dtype=int and shape (,N)
                         the states for which there exist non-zero matrix elements for given state
         matrixelements: float
                         the non-zero matrix elements
    """

    N=len(state)
    sigma = np.copy(state)
    for i in range(1, N-1):
        sigma[i] = np.abs(1 - sigma[i])
    el = 1
    return np.array(sigma), el
    

def DeltaX_average (N, gs, periodic= False):
    basis = []

    #Generate a basis
    for i in range(2**N):
        basis_temp = np.zeros((N))
        a = np.array([int(d) for d in bin(i)[2:]])
        l = len(a)
        basis_temp[N-l:] = a

        basis.append(basis_temp)

    basis = np.array(basis)

    delta_x = 0.0
    for i in range(basis.shape[0]):
        delta_s, delta_el = DeltaX_MatrixElement(np.reshape(basis[i,:],(N)), periodic)
        for j in range(basis.shape[0]):
            if np.all(basis[j,:]==delta_s):
                delta_x += delta_el*gs[i]*gs[j]
    return delta_x




def DeltaY_MatrixElement( state, periodic = False):
    """
    Given "state" returns the only non-zero state when the DeltaX operator is applied
    Uses the following mapping:
    1-j1-2-j2-5-j1-6
    |    |    |    |
    0-j2-3-j1-4-j2-7
    DeltaX is defined as PI_i=1^2j Sx_i
    ---------------------------------------------------------------------------------------
    Returns: 2-tuple of type (np.ndarray,np.ndarray)
         sigmas:         np.ndarray of dtype=int and shape (,N)
                         the states for which there exist non-zero matrix elements for given state
         matrixelements: float
                         the non-zero matrix elements
    """

    N=len(state)
    sigma = np.copy(state)
    sigma[0] = np.abs(1 - sigma[0])
    for i in range(2, N-2):
        sigma[i] = np.abs(1 - sigma[i])
    sigma[-1] = np.abs(1 - sigma[-1])
    el = 1
    return np.array(sigma), el
    

def DeltaY_average (N, gs, periodic= False):
    basis = []

    #Generate a basis
    for i in range(2**N):
        basis_temp = np.zeros((N))
        a = np.array([int(d) for d in bin(i)[2:]])
        l = len(a)
        basis_temp[N-l:] = a

        basis.append(basis_temp)

    basis = np.array(basis)

    delta_y = 0.0
    for i in range(basis.shape[0]):
        delta_s, delta_el = DeltaY_MatrixElement(np.reshape(basis[i,:],(N)), periodic)
        for j in range(basis.shape[0]):
            if np.all(basis[j,:]==delta_s):
                delta_y += delta_el*gs[i]*gs[j]
    return delta_y




def DeltaX_RungMatrixElement( state, periodic = False):
    """
    Given "state" returns the only non-zero state when the DeltaX operator is applied
    Uses the following mapping:
    1-j1-2-j2-5-j1-6
    |    |    |    |
    0-j2-3-j1-4-j2-7
    DeltaX is defined as PI_i=1^2j Sx_i
    ---------------------------------------------------------------------------------------
    Returns: 2-tuple of type (np.ndarray,np.ndarray)
         sigmas:         np.ndarray of dtype=int and shape (,N)
                         the states for which there exist non-zero matrix elements for given state
         matrixelements: float
                         the non-zero matrix elements
    """

    N=len(state)
    sigma = np.copy(state)
    if sigma[0]==0:
        sigma[0]=2
    elif sigma[0]==1:
        sigma[0]=3
    elif sigma[0]==2:
        sigma[0]=0
    elif sigma[0]==3:
        sigma[0]=1
    
    if N%2 ==0:
        if sigma[-1]==0:
            sigma[-1]=2
        elif sigma[-1]==1:
            sigma[-1]=3
        elif sigma[-1]==2:
            sigma[-1]=0
        elif sigma[-1]==3:
            sigma[-1]=1
    else:
        if sigma[-1]==0:
            sigma[-1]=1
        elif sigma[-1]==1:
            sigma[-1]=0
        elif sigma[-1]==2:
            sigma[-1]=3
        elif sigma[-1]==3:
            sigma[-1]=2


    for i in range(1, N-1):
        if sigma[i]==0:
            sigma[i]=3
        elif sigma[i]==1:
            sigma[i]=2
        elif sigma[i]==2:
            sigma[i]=1
        elif sigma[i]==3:
            sigma[i]=0
    el = 1
    return np.array(sigma), el
    

def DeltaX_RungAverage (N, gs, periodic= False):
    basis = []

    #Generate a basis
    for i in range(4**N):
        basis_temp = np.zeros((N))
        a = np.array([int(d) for d in np.base_repr(i, base=4)])
        l = len(a)
        basis_temp[N-l:] = a

        basis.append(basis_temp)

    basis = np.array(basis)

    delta_x = 0.0
    for i in range(basis.shape[0]):
        delta_s, delta_el = DeltaX_RungMatrixElement(np.reshape(basis[i,:],(N)), periodic)
        for j in range(basis.shape[0]):
            if np.all(basis[j,:]==delta_s):
                delta_x += delta_el*gs[i]*gs[j]
    return delta_x




def DeltaY_RungMatrixElement( state, periodic = False):
    """
    Given "state" returns the only non-zero state when the DeltaX operator is applied
    Uses the following mapping:
    1-j1-2-j2-5-j1-6
    |    |    |    |
    0-j2-3-j1-4-j2-7
    DeltaX is defined as PI_i=1^2j Sx_i
    ---------------------------------------------------------------------------------------
    Returns: 2-tuple of type (np.ndarray,np.ndarray)
         sigmas:         np.ndarray of dtype=int and shape (,N)
                         the states for which there exist non-zero matrix elements for given state
         matrixelements: float
                         the non-zero matrix elements
    """

    N=len(state)
    sigma = np.copy(state)
    sign = 1

    if sigma[0]==0:
        sigma[0]=1
    elif sigma[0]==1:
        sigma[0]=0
        sign *=-1
    elif sigma[0]==2:
        sigma[0]=3
    elif sigma[0]==3:
        sigma[0]=2
        sign *=-1

    if N%2 ==0:
        if sigma[-1]==0:
            sigma[-1]=1
        elif sigma[-1]==1:
            sigma[-1]=0
            sign *=-1
        elif sigma[-1]==2:
            sigma[-1]=3
        elif sigma[-1]==3:
            sigma[-1]=2
            sign *=-1
    else:
        if sigma[-1]==0:
            sigma[-1]=2
        elif sigma[-1]==1:
            sigma[-1]=3
        elif sigma[-1]==2:
            sigma[-1]=0
            sign *=-1
        elif sigma[-1]==3:
            sigma[-1]=1
            sign *=-1


    for i in range(1, N-1):
        if sigma[i]==0:
            sigma[i]=3
        elif sigma[i]==1:
            sigma[i]=2
            sign*=-1
        elif sigma[i]==2:
            sigma[i]=1
            sign*=-1
        elif sigma[i]==3:
            sigma[i]=0
    el = sign * (-1)**(N-1)
    return np.array(sigma), el
    

def DeltaY_RungAverage (N, gs, periodic= False):
    basis = []

    #Generate a basis
    for i in range(4**N):
        basis_temp = np.zeros((N))
        a = np.array([int(d) for d in np.base_repr(i, base=4)])
        l = len(a)
        basis_temp[N-l:] = a

        basis.append(basis_temp)

    basis = np.array(basis)

    delta_y = 0.0
    for i in range(basis.shape[0]):
        delta_s, delta_el = DeltaY_RungMatrixElement(np.reshape(basis[i,:],(N)), periodic)
        for j in range(basis.shape[0]):
            if np.all(basis[j,:]==delta_s):
                delta_y += delta_el*gs[i]*gs[j]
    return delta_y


def energy_rung_average (N, gs, j1, j2, j3, periodic= False):
    basis = []

    #Generate a basis
    for i in range(4**N):
        basis_temp = np.zeros((N))
        a = np.array([int(d) for d in np.base_repr(i, base=4)])
        l = len(a)
        basis_temp[N-l:] = a

        basis.append(basis_temp)

    basis = np.array(basis)

    nrg = 0.0
    for n in range(basis.shape[0]):
        sigmas,elements= KitaevRungMatrixElements(j1, j2, j3,np.reshape(basis[n,:],(N)),periodic)
        for m in range(sigmas.shape[0]):
            for b in range(basis.shape[0]):
                if np.all(basis[b,:]==sigmas[m,:]):
                    nrg +=elements[m] * gs[b]*gs[n]
                    break
    return nrg