from operator import setitem
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #stop displaying tensorflow warnings
import numpy as np
import os
import random
from math import ceil

from ComplexRNNwavefunction import RNNwavefunction

from RNNwavefunction_paritysym import SymRNNwavefunction #To use an RNN that has a parity symmetry so that the RNN is not biased by autoregressive sampling from left to right
from RNNwavefunction_zSpinInversionSym import SpinFlipSymRNNwavefunction
# Loading Functions --------------------------
def Kitaev_local_energies(J1, J2, J3, samples, queue_samples, log_ampl_tensor, samples_placeholder, log_amplitudes, batch_size, sess):
    """
    To get the local energies of  2-leg ladder Kitaev's model (OBC), as a 1D chain given a set of set of samples in parallel!
    Returns: (numsamples) The local energies that correspond to the "samples"
    Inputs:
    - samples: (numsamples, N)
    - J1: float
    - J2: float
    - J3: float
    - queue_samples: ((2(N-1)+1), numsamples, N) an empty allocated np array to store the non diagonal elements (+ the diagonal)
    - log_ampl_tensor: A TF tensor with size (None)
    - samples_placeholder: A TF placeholder to feed in a set of configurations
    - log_amplitudes: ((2N-1)*numsamples) an empty allocated np array to store the log_amplitudes of non diagonal elements (+ the diagonal)
    - sess: The current TF session
    """
    numsamples = samples.shape[0]
    N = samples.shape[1]

    local_energies = np.zeros((numsamples), dtype = np.complex64)

    #Diagonal 
    for i in range(N):
        valuesT = np.copy(samples[:,i])
        valuesT[samples[:,i]==1] = -1 #Opposite spin
        valuesT[samples[:,i]==2] = -1 #Opposite spin
        valuesT[samples[:,i]==3] = +1 #Same spin
        valuesT[samples[:,i]==0] = +1 #Same spin
        local_energies += valuesT*(J3)

    #Storing the diagonal samples
    queue_samples[0] = samples 

    #Non-diagonal
    y_states_sign_map = []
    if J1 != 0 or J2 !=0:
        for i in range(N-1):

            #Even   #SxSx
                    #SySy
            if (i%2==0):

                #Flip upper spin i and i+1 --> upper spin flip = +2 with 4 = 0
                x_interaction_states = np.copy(samples)
                x_interaction_states[:,i] = (2 + x_interaction_states[:,i]) % 4
                x_interaction_states[:,i+1] = (2 + x_interaction_states[:,i+1]) % 4

                #Always remember x-interactions in odd index
                queue_samples[2*i+1] = x_interaction_states 
                
                #Flip lower spin i and i+1 --> lower spin flip = +1 if even, else -1
                # 1 - 2* ([2k, 2k+1] % 2) --> 1 - 2*([0,1]) --> [1, -1]
                y_interaction_states = np.copy(samples)
                y_interaction_states[:,i] += 1 - 2 * (y_interaction_states[:,i] % 2)
                y_interaction_states[:,i+1] += 1 - 2 * (y_interaction_states[:,i+1] % 2)

                #For lower spin y interaction is negative if the sum of the 2 new states is even
                #Remember the sign of each of these states
                # 1 - 2*[T,F] --> [-1, +1]
                sign_map = 1 - 2 * ( (y_interaction_states[:, i] + y_interaction_states[:, i+1])%2 == 0 )
                y_states_sign_map.append(sign_map)

                #Always remember y-interactions in even index
                queue_samples[2*i+2] = y_interaction_states


            #Odd    #SySy
                    #SxSx
            else:

                #Flip upper spin i and i+1 --> upper spin flip = +2 with 4 = 0
                y_interaction_states = np.copy(samples)
                y_interaction_states[:,i] = (2 + y_interaction_states[:,i]) % 4
                y_interaction_states[:,i+1] = (2 + y_interaction_states[:,i+1]) % 4
                
                #For upper spin y interaction is negative if the 2 new states are the same, or if their sum%4 is 1 (0,1/2,3)
                #Remember the sign of each of these states
                # 1 - 2*[T,F] --> [-1, +1]
                sign_map = 1 - 2 * ( (y_interaction_states[:, i] == y_interaction_states[:, i+1]) |  ((y_interaction_states[:, i] + y_interaction_states[:, i+1])%4 == 1) )
                y_states_sign_map.append(sign_map)

                #Always remember y-interactions in even index
                queue_samples[2*i+2] = y_interaction_states
                
                #Flip lower spin i and i+1 --> lower spin flip = +1 if even, else -1
                # 1 - 2* ([2k, 2k+1] % 2) --> 1 - 2*([0,1]) --> [1, -1]
                x_interaction_states = np.copy(samples)
                x_interaction_states[:,i] += 1 - 2 * (x_interaction_states[:,i] % 2)
                x_interaction_states[:,i+1] += 1 - 2 * (x_interaction_states[:,i+1] % 2)

                #Always remember x-interactions in odd index
                queue_samples[2*i+1] = x_interaction_states
                
                
    #Calculating log_amplitudes from samples
    #Do it in steps

    #Number of total configurations
    len_sigmas = (2*N-1)*numsamples
    steps = ceil(len_sigmas/batch_size) #Get a maximum of batch_size configurations in batch size just to not allocate too much memory

    #Reshape samples
    queue_samples_reshaped = np.reshape(queue_samples, [(2*N-1)*numsamples, N]) #from  ( (2N-1), numsamples, N )

    for i in range(steps):
        if i < steps-1:
            cut = slice((i*len_sigmas)//steps,((i+1)*len_sigmas)//steps)      #in this way batches are equally balanced (if 48k tot config - > 24k, 24k, not 25, 23)
        else:
            cut = slice((i*len_sigmas)//steps,len_sigmas)
        
        log_amplitudes[cut] = sess.run(log_ampl_tensor, feed_dict={samples_placeholder:queue_samples_reshaped[cut]})


    log_ampl_reshaped = np.reshape(log_amplitudes, [ (2*N-1) ,numsamples])    #it was ((N+1)*numsamples)

    #Sum x interactions (odd indices)
    local_energies += J1*np.sum(np.exp(log_ampl_reshaped[1::2,:]-log_ampl_reshaped[0,:]), axis = 0)

    print("x = ", np.exp(log_ampl_reshaped[1::2,:]-log_ampl_reshaped[0,:]))

    #Sum y-interactions (even indices) with the correct sign
    local_energies += J2*np.sum(np.array(y_states_sign_map) * np.exp(log_ampl_reshaped[2::2,:]-log_ampl_reshaped[0,:]), axis = 0)

    print("y = ", np.exp(log_ampl_reshaped[2::2,:]-log_ampl_reshaped[0,:]))
    return local_energies
#--------------------------


# ---------------- Running VMC with RNNs -------------------------------------
def run_Kita1D(numsteps = 10**4, systemsize = 20, num_units = 50, J1 = 1, J2 = 1, J3=1, 
                num_layers = 1, numsamples = 200, learningrate = 5e-3, batch_size = 25000, seed = 111, 
                symRNN = 'None', load_model = False,
                save_dir = ".", checkpoint_steps = 11000):

    #Seeding ---------------------------------------------
    tf.reset_default_graph()
    random.seed(seed)  # `python` built-in pseudo-random generator
    np.random.seed(seed)  # numpy pseudo-random generator
    tf.set_random_seed(seed)  # tensorflow pseudo-random generator

    #End Seeding ---------------------------------------------

    # System size
    N = systemsize

    #Learning rate
    lr=np.float64(learningrate)

    # Intitializing the RNN-----------
    units=[num_units]*num_layers #list containing the number of hidden units for each layer of the networks

    input_dim=4 #Dimension of the Hilbert space for each site (here = 4, up or down)
    numsamples_=20 #only for initialization; later I'll use a much larger value (see below)

    
    if (symRNN in ['None', 'none']): 
        wf=RNNwavefunction(N, inputdim=4, units=units,cell=tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell, seed = seed) #contains the graph with the RNNs
    elif (symRNN == 'parity'):
        wf=SymRNNwavefunction(N,units=units,cell=tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell, seed = seed)
    elif (symRNN == 'spinflip'):
        wf=SpinFlipSymRNNwavefunction(N,units=units,cell=tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell, seed = seed)
    else:
        print("Unknown symmetry: ", symRNN)
        print("Accepted values are: 'None'/'none', 'parity' or 'spinflip'")
        return -1, -1
        
    
    sampling=wf.sample(numsamples_,input_dim) #call this function once to create the dense layers

    #Initialize everything --------------------
    with wf.graph.as_default():
        samples_placeholder=tf.placeholder(dtype=tf.int32,shape=[numsamples_,N]) #the samples_placeholder are the samples of all of the spins
        global_step = tf.Variable(0, trainable=False)
        learningrate_placeholder=tf.placeholder(dtype=tf.float64,shape=[])
        learning_rate_withexpdecay = tf.train.exponential_decay(learningrate_placeholder, global_step = global_step, decay_steps = 100, decay_rate = 1.0, staircase=True) #For exponential decay of the learning rate (only works if decay_rate < 1.0)
        ampl=wf.log_amplitude(samples_placeholder,input_dim) #The probs are obtained by feeding the sample of spins.
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate_withexpdecay) #Using AdamOptimizer
        init=tf.global_variables_initializer()
    # End Intitializing ----------------------------

    #Starting Session------------
    #Activating GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess=tf.Session(graph=wf.graph, config=config)
    sess.run(init)
    #---------------------------

    #Counting the number of parameters
    with wf.graph.as_default():
        variables_names =[v.name for v in tf.trainable_variables()]
        n_parameters = 0
        values = sess.run(variables_names)
        for k,v in zip(variables_names, values):
            v1 = tf.reshape(v,[-1])
            print(k,v1.shape)
            n_parameters +=v1.shape[0]
        print('The number of params is {0}'.format(n_parameters))

    #Building the graph -------------------

    path=os.getcwd()

    ending='_units'
    for u in units:
        ending+='_{0}'.format(u)
    
    if (symRNN not in ['None', 'none']):
        ending += '_' + symRNN + 'Symm'
    
    param_string = 'N'+str(N)+'_samp'+str(numsamples)+'_J1'+str(J1)+'_J2'+str(J2)+'_J3'+str(J3)+'_GRURNN_OBC'
    
    filename= save_dir + '/RNNwavefunction_' + param_string + ending + '.ckpt'
    savename = '_KITA'

    with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
        with wf.graph.as_default():
            Eloc=tf.placeholder(dtype=tf.complex64,shape=[numsamples])
            samp=tf.placeholder(dtype=tf.int32,shape=[numsamples,N])
            log_amplitudes_=wf.log_amplitude(samp,inputdim=input_dim)

            #Now calculate the fake cost function: https://stackoverflow.com/questions/33727935/how-to-use-stop-gradient-in-tensorflow
            #stop_gradient prevents the optimization of Eloc as a variable(?)
            cost = 2*tf.real(tf.reduce_mean(tf.conj(log_amplitudes_)*tf.stop_gradient(Eloc)) - tf.conj(tf.reduce_mean(log_amplitudes_))*tf.reduce_mean(tf.stop_gradient(Eloc)))
            #Calculate Gradients---------------

            gradients, variables = zip(*optimizer.compute_gradients(cost))

            #End calculate Gradients---------------

            optstep=optimizer.apply_gradients(zip(gradients,variables),global_step=global_step)
            sess.run(tf.variables_initializer(optimizer.variables()))
            saver=tf.train.Saver(max_to_keep=3)
    #----------------------------------------------------------------

    meanEnergy=[]
    varEnergy=[]

    #Loading previous trainings (uncomment if you wanna restore a previous session)----------
    if load_model:
        path=os.getcwd()
        ending='_units'
        for u in units:
            ending+='_{0}'.format(u)
        savename = '_TFIM'
        with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
            with wf.graph.as_default():
                saver.restore(sess,path+'/'+filename)
                meanEnergy=np.load(save_dir + '/meanEnergy_' + param_string + savename + ending + '.npy').tolist()
                varEnergy=np.load(save_dir + '/varEnergy_'+ param_string + savename + ending + '.npy').tolist()
    #------------------------------------

    with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
        with wf.graph.as_default():


            samples_ = wf.sample(numsamples=numsamples,inputdim=input_dim)
            samples = np.ones((numsamples, N), dtype=np.int32)  
            samples_placeholder=tf.placeholder(dtype=tf.int32,shape=(None,N))
            log_ampl_tensor=wf.log_amplitude(samples_placeholder,inputdim=input_dim)

        #Allocate array to store matrix elements and log_amplitudes
        #Do this here for memory efficiency as we do not want to allocate it at each training step
            queue_samples = np.zeros((2*N-1, numsamples, N), dtype = np.int32)
            log_amplitudes = np.zeros((2*N-1)*numsamples, dtype=np.complex64) 

            for it in range(len(meanEnergy),numsteps+1):

                samples=sess.run(samples_)

                #Estimating local_energies
                local_energies = Kitaev_local_energies(J1, J2, J3, samples, queue_samples, log_ampl_tensor, samples_placeholder, log_amplitudes, batch_size, sess)

                meanE = np.mean(local_energies)
                varE = np.var(np.real(local_energies))

                #adding elements to be saved
                meanEnergy.append(meanE)
                varEnergy.append(varE)

                sess.run(optstep,feed_dict={Eloc:local_energies,samp:samples,learningrate_placeholder: lr})

                #Comment if you don't want to save
                if it%checkpoint_steps==0:
                    print("Step: ", it)
                    #Saving the performances
                    np.save(save_dir + '/meanEnergy_'+ param_string + savename + ending + '.npy',meanEnergy)
                    np.save(save_dir + '/varEnergy_'+ param_string + savename + ending + '.npy',varEnergy)
                    #Saving the model 
                    saver.save(sess,path+'/'+filename, global_step = it)
        
        if (numsteps%checkpoint_steps!=0):
            #Saving the performances
            np.save(save_dir + '/meanEnergy_'+ param_string + savename + ending + '.npy',meanEnergy)
            np.save(save_dir + '/varEnergy_'+ param_string + savename + ending + '.npy',varEnergy)
            #Saving the model 
            saver.save(sess,path+'/'+filename, global_step = it)
 
    return meanEnergy, varEnergy
    #-------------------------------------------------------------------------------------------


def sample_from_model(numsamples = 10**6, old_numsamples = 200, 
                      systemsize = 20, J1=1, J2=1, J3=2, 
                      num_units = 50, num_layers = 1, 
                      seed = 111, symRNN = False,
                      save_dir = ".", model_step = None):

    #Seeding ---------------------------------------------
    tf.reset_default_graph()
    random.seed(seed)  # `python` built-in pseudo-random generator
    np.random.seed(seed)  # numpy pseudo-random generator
    tf.set_random_seed(seed)  # tensorflow pseudo-random generator
    #End Seeding ---------------------------------------------
    
    # System size
    N = systemsize

    # Intitializing the RNN-----------
    units=[num_units]*num_layers #list containing the number of hidden units for each layer of the networks


    if (symRNN in ['None', 'none']): 
        wf=RNNwavefunction(N, inputdim=4, units=units,cell=tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell, seed = seed) #contains the graph with the RNNs
    elif (symRNN == 'parity'):
        wf=SymRNNwavefunction(N,units=units,cell=tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell, seed = seed)
    elif (symRNN == 'spinflip'):
        wf=SpinFlipSymRNNwavefunction(N,units=units,cell=tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell, seed = seed)
    else:
        print("Unknown symmetry: ", symRNN)
        print("Accepted values are: 'None'/'none', 'parity' or 'spinflip'")
        return -1

    numsamples_ = 20
    sampling=wf.sample(numsamples_,inputdim = 4) #call this function once to create the dense layers

    #Initialize everything --------------------
    with wf.graph.as_default():
        samples_placeholder=tf.placeholder(dtype=tf.int32,shape=[numsamples_,N]) #the samples_placeholder are the samples of all of the spins
        init=tf.global_variables_initializer()
    # End Intitializing ----------------------------

    #Starting Session------------
    #Activating GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess=tf.Session(graph=wf.graph, config=config)
    sess.run(init)
    #----------------------------
    #Setting filename------------
    path=os.getcwd()
    ending='_units'
    for u in units:
        ending+='_{0}'.format(u)
    
    if (symRNN not in ['None', 'none']):
        ending += '_' + symRNN + 'Symm'
    if model_step != None:
        steps_string = '-{}'.format(model_step)
    
    param_string = 'N'+str(N)+'_samp'+str(old_numsamples)+'_J1'+str(J1)+'_J2'+str(J2)+'_J3'+str(J3)+'_GRURNN_OBC'
    
    filename= save_dir + '/RNNwavefunction_' + param_string + ending + '.ckpt' + steps_string
    savename = '_KITA'
    #-----------------------------
    #Restoring old session
    #and computing quantities
    

    with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
        with wf.graph.as_default():
            saver=tf.train.Saver()
            saver.restore(sess,path+'/'+filename)

            samples_ = wf.sample(numsamples=numsamples,inputdim=4)
            samples=sess.run(samples_)
            np.save(save_dir + '/samples_'+param_string + savename + ending + '.npy',samples)
            
    return samples




##################
## NEW WAY
##################
def KitaevMatrixElements(j1,j2,j3,sigmap, sigmaH, matrixelements, periodic = False):
    """
    -Computes the matrix element of the Kitaev model for a given configuration sigmap
    -We hope to make this function parallel in future versions to return the matrix elements of a large number of configurations
    -----------------------------------------------------------------------------------
    Parameters:
    j1, j2, j3: np.ndarray of shape (N), (N) and (N), respectively, and dtype=float:
                J1J2 parameters
    sigmap:     np.ndarrray of dtype=int and shape (N)
                rung-state, integer encoded (using 0 for dd, 1 du, 2 ud, 3 uu)
                A sample of spins can be fed here.
    sigmaH: an array to store the diagonal and the diagonal configurations after applying the Hamiltonian on sigmap.
    matrixelements: an array where to store the matrix elements after applying the Hamiltonian on sigmap.
    periodic: bool, indicate if the chain is periodic on not.
    -----------------------------------------------------------------------------------
    Returns: num, float which indicate the number of diagonal and non-diagonal configurations after applying the Hamiltonian on sigmap
    """
    N=len(j1)

    num = 0 #Number of basis elements

    if periodic:
        limit = N + 1
    else:
        limit = N   
    #Diagonal interaction (SzSz)
    diag = 0

    for site in range(limit):
        if sigmap[site]== 1 or sigmap[site] == 2: #if the two neighouring spins are opposite
            diag-=0.25*j3[site] #add a negative energy contribution
        else:
            diag+=0.25*j3[site]
    
    matrixelements[num] = diag #add the diagonal part to the matrix elements

    sig = np.copy(sigmap)

    sigmaH[num] = sig

    num += 1

    #off-diagonal part:
    for site in range(limit - 1):
    
        if (site%2==0):
            #Flip spin 2j and 2j-1
            if j1[site] != 0.0:
                sig=np.copy(sigmap)
                sig[site]  = (2 + sigmap[site]) % 4
                sig[site +1]  = (2 + sigmap[site +1]) % 4
                sigmaH[num] = sig
                matrixelements[num] = j1[site] * 0.25
                num += 1
            if j2[site] != 0.0:
                sig=np.copy(sigmap)
                sig[site] += 1 - 2 * (sig[site] % 2)
                sig[site+1] += 1 - 2 * (sig[site+1] % 2)
                sigmaH[num] = sig
                sign = 1 - 2 * ( (sig[site] + sig[site+1])%2 == 0 )
                matrixelements[num] =  sign * j2 * 0.25
                num += 1
        
        else:
            if j2[site] != 0.0:
                sig=np.copy(sigmap)
                sig[site]  = (2 + sigmap[site]) % 4
                sig[site +1]  = (2 + sigmap[site +1]) % 4
                sigmaH[num] = sig
                sign = 1 - 2 * ( (sig[site] == sig[site+1]) |  ((sig[site] + sig[site+1])%4 == 1) )
                matrixelements[num] =  sign * j2 * 0.25
                num += 1
            
            if j1[site] != 0.0:
                sig=np.copy(sigmap)
                sig[site] += 1 - 2 * (sig[site] % 2)
                sig[site+1] += 1 - 2 * (sig[site+1] % 2)
                sigmaH[num] = sig
                matrixelements[num] = j1[site] * 0.25
                num += 1

    return num

def J1J2Slices(J1, J2, Bz, sigmasp, sigmas, H, sigmaH, matrixelements, Marshall_sign):
    """
    Returns: A tuple -The list of slices (that will help to slice the array sigmas)
             -Total number of configurations after applying the Hamiltonian on the list of samples sigmasp (This will be useful later during training, note that it is not constant for J1J2 as opposed to TFIM)
    ----------------------------------------------------------------------------
    Parameters:
    J1, J2, Bz: np.ndarray of shape (N), (N) and (N), respectively, and dtype=float:
                J1J2 parameters
    sigmasp:    np.ndarrray of dtype=int and shape (numsamples,N)
                spin-states, integer encoded (using 0 for down spin and 1 for up spin)
    sigmas: an array to store the diagonal and the diagonal configurations after applying the Hamiltonian on all the samples sigmasp.
    H: an array to store the diagonal and the diagonal matrix elements after applying the Hamiltonian on all the samples sigmasp.
    sigmaH: an array to store the diagonal and the diagonal configurations after applying the Hamiltonian on a single sample.
    matrixelements: an array where to store the matrix elements after applying the Hamiltonian on sigmap on a single sample.
    Marshall_sign: bool, indicate if the Marshall sign is applied or not.    
    ----------------------------------------------------------------------------
    """

    slices=[]
    sigmas_length = 0

    for n in range(sigmasp.shape[0]):
        sigmap=sigmasp[n,:]
        num = J1J2MatrixElements(J1,J2,Bz,sigmap, sigmaH, matrixelements, Marshall_sign)#note that sigmas[0,:]==sigmap, matrixelements and sigmaH are updated
        slices.append(slice(sigmas_length,sigmas_length + num))
        s = slices[n]

        H[s] = matrixelements[:num]
        sigmas[s] = sigmaH[:num]

        sigmas_length += num #Increasing the length of matrix elements sigmas

    return slices, sigmas_length