import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #stop displaying tensorflow warnings
import numpy as np
import os
import time
import random
from math import ceil

from RNNwavefunction import RNNwavefunction
from RNNwavefunction_paritysym import SymRNNwavefunction #To use an RNN that has a parity symmetry so that the RNN is not biased by autoregressive sampling from left to right
from RNNwavefunction_zSpinInversionSym import SpinFlipSymRNNwavefunction
# Loading Functions --------------------------
def Ising_local_energies(Jz, Bx, samples, queue_samples, log_probs_tensor, samples_placeholder, log_probs, batch_size, sess):
    """ To get the local energies of 1D TFIM (OBC) given a set of set of samples in parallel!
    Returns: The local energies that correspond to the "samples"
    Inputs:
    - samples: (numsamples, N)
    - Jz: (N) np array
    - Bx: float
    - queue_samples: ((N+1)*numsamples, N) an empty allocated np array to store the non diagonal elements
    - log_probs_tensor: A TF tensor with size (None)
    - samples_placeholder: A TF placeholder to feed in a set of configurations
    - log_probs: ((N+1)*numsamples) an empty allocated np array to store the log_probs non diagonal elements
    - sess: The current TF session
    """
    numsamples = samples.shape[0]
    N = samples.shape[1]

    local_energies = np.zeros((numsamples), dtype = np.float64)

    for i in range(N-1): #diagonal elements
        values = samples[:,i]+samples[:,i+1]
        valuesT = np.copy(values)
        valuesT[values==2] = +1 #If both spins are up
        valuesT[values==0] = +1 #If both spins are down
        valuesT[values==1] = -1 #If they are opposite
        local_energies += valuesT*(-Jz[i])

    queue_samples[0] = samples #storing the diagonal samples

    if Bx != 0:
        for i in range(N):  #Non-diagonal elements
            valuesT = np.copy(samples)
            valuesT[:,i][samples[:,i]==1] = 0 #Flip to 0 spin i of each sampling, if in the original was 1
            valuesT[:,i][samples[:,i]==0] = 1 #Flip to 1 spin i of each sampling, if in the original was 0

            queue_samples[i+1] = valuesT

    #Calculating log_probs from samples
    #Do it in steps

    # print("Estimating log probs started")
    # start = time.time()

    len_sigmas = (N+1)*numsamples
    steps = ceil(len_sigmas/batch_size) #Get a maximum of 25000 configurations in batch size just to not allocate too much memory

    queue_samples_reshaped = np.reshape(queue_samples, [(N+1)*numsamples, N]) #cause it was ((N+1), numsamples, N)
    for i in range(steps):
      if i < steps-1:
          cut = slice((i*len_sigmas)//steps,((i+1)*len_sigmas)//steps)      #in this way batches are equally balanced (if 48k tot config - > 24k, 24k, not 25, 23)
      else:
          cut = slice((i*len_sigmas)//steps,len_sigmas)
      log_probs[cut] = sess.run(log_probs_tensor, feed_dict={samples_placeholder:queue_samples_reshaped[cut]})

    # end = time.time()
    # print("Estimating log probs ended ", end-start)

    log_probs_reshaped = np.reshape(log_probs, [N+1,numsamples]) #it was ((N+1)*numsamples)
    
#     for j in range(numsamples):
#         local_energies[j] += -Bx*np.sum(np.exp(0.5*log_probs_reshaped[1:,j]-0.5*log_probs_reshaped[0,j]))
    local_energies += -Bx*np.sum(np.exp(0.5*log_probs_reshaped[1:,:]-0.5*log_probs_reshaped[0,:]), axis = 0) #This is faster than previous loop, since it runs in parallel
    return local_energies
#--------------------------

#-------------Spins correlations-----------------

#--------------------SzSz--------------------------
def Ising_SzSz_correlations(samples):
    """
    Computing the Sz^iSz^j correlations with i = the spin in the middle (the most similar to bulk conditions)
    --------------------------------------------------------------------------------------------------------
    Parameters:
    - samples: (numsamples, N)
    --------------------------------------------------------------------------------------------------------
    Returns: ndarray of size (N/2, numsamples) with the vector of correlations (numsamples value) between the spin N/2 and the (N/2 + i)-esim spin in position i 
    """
    numsamples = samples.shape[0]
    N = samples.shape[1]
    middle_spin = ceil(N/2)
    Sz_correlations = np.zeros( (N - middle_spin, numsamples), dtype = np.float64)

    for i in range(middle_spin, N): #only diagonal elements are of interest
        values = samples[:,middle_spin -1]+samples[:,i]
        valuesT = np.copy(values)
        valuesT[values==2] = +1 #If both spins are up
        valuesT[values==0] = +1 #If both spins are down
        valuesT[values==1] = -1 #If they are opposite
        Sz_correlations[i - middle_spin] = 0.25 * valuesT

    return Sz_correlations
#---------------------------------------------------

#--------------------SxSx---------------------------
def Ising_SxSx_correlations (samples, queue_samples, log_probs_tensor, samples_placeholder, log_probs, batch_size, sess):
    """
    Computing the Sx^iSx^j correlations
    --------------------------------------------------------------------------------------------------------
    Parameters:
    - samples: (numsamples, N)
    - queue_samples: ((N+1)*numsamples, N) an empty allocated np array to store the non diagonal elements
    - log_probs_tensor: A TF tensor with size (None)
    - samples_placeholder: A TF placeholder to feed in a set of configurations
    - log_probs: ((N+1)*numsamples) an empty allocated np array to store the log_probs non diagonal elements
    - sess: The current TF session
    --------------------------------------------------------------------------------------------------------
    Returns: ndarray of size (N/2, numsamples) with the vector of correlations (numsamples value) between the spin N/2 and the (N/2 + i)-esim spin in position i 
    """
    numsamples = samples.shape[0]
    N = samples.shape[1]
    middle_spin = ceil(N/2)
    
    #store the diagonal samples -> needed for the evaluation of matrix element
    queue_samples[0] = samples 

    #flip spin from which we compute the correlation
    flipped_samples = np.copy(samples)
    flipped_samples[:, middle_spin -1][samples[:, middle_spin -1]==1] = 0
    flipped_samples[:, middle_spin -1][samples[:, middle_spin -1]==0] = 1
    queue_samples[middle_spin] = flipped_samples
    
    for i in range(middle_spin, N):  #Non-diagonal elements
        valuesT = np.copy(flipped_samples)
        valuesT[:,i][flipped_samples[:,i]==1] = 0 #Flip to 0 spin i of each sampling, if in the original was 1
        valuesT[:,i][flipped_samples[:,i]==0] = 1 #Flip to 1 spin i of each sampling, if in the original was 0
        queue_samples[i+1] = valuesT

    #Calculating log_probs from samples in steps
    len_sigmas = (N+1)*numsamples
    steps = ceil(len_sigmas/batch_size) #Get a maximum of 25000 configurations in batch size just to not allocate too much memory

    queue_samples_reshaped = np.reshape(queue_samples, [(N+1)*numsamples, N]) #cause it was ((N+1), numsamples, N)

    for i in range(steps):
        if i < steps-1:
            cut = slice((i*len_sigmas)//steps,((i+1)*len_sigmas)//steps)      #in this way batches are equally balanced (if 48k tot config - > 24k, 24k, not 25, 23)
        else:
            cut = slice((i*len_sigmas)//steps,len_sigmas)
        log_probs[cut] = sess.run(log_probs_tensor, feed_dict={samples_placeholder:queue_samples_reshaped[cut]})

    log_probs_reshaped = np.reshape(log_probs, [N+1,numsamples]) #it was ( (N+1)*numsamples )
    
    Sx_correlations = 0.25 * np.exp(0.5*log_probs_reshaped[middle_spin+1:,:]-0.5*log_probs_reshaped[0,:])
    return Sx_correlations
#---------------------------------------------------

# ---------------- Running VMC with RNNs -------------------------------------
def run_1DTFIM(numsteps = 10**4, systemsize = 20, num_units = 50, Bx = 1, 
                num_layers = 1, numsamples = 500, learningrate = 5e-3, batch_size = 25000, seed = 111, 
                symRNN = 'None', load_model = False,
                save_dir = "../Check_Points/1DTFIM", checkpoint_steps = 500):

    #Seeding ---------------------------------------------
    tf.reset_default_graph()
    random.seed(seed)  # `python` built-in pseudo-random generator
    np.random.seed(seed)  # numpy pseudo-random generator
    tf.set_random_seed(seed)  # tensorflow pseudo-random generator

    #End Seeding ---------------------------------------------

    # System size
    N = systemsize
    
    Jz = +np.ones(N) #Ferromagnetic coupling

    #Learning rate
    lr=np.float64(learningrate)

    # Intitializing the RNN-----------
    units=[num_units]*num_layers #list containing the number of hidden units for each layer of the networks

    input_dim=2 #Dimension of the Hilbert space for each site (here = 2, up or down)
    numsamples_=20 #only for initialization; later I'll use a much larger value (see below)

    
    if (symRNN in ['None', 'none']): 
        wf=RNNwavefunction(N,units=units,cell=tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell, seed = seed) #contains the graph with the RNNs
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
        probs=wf.log_probability(samples_placeholder,input_dim) #The probs are obtained by feeding the sample of spins.
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
     
    filename= save_dir + '/RNNwavefunction_N'+str(N)+'_samp'+str(numsamples)+'_Jz1Bx'+str(Bx)+'_GRURNN_OBC'+ending + '.ckpt'
    savename = '_TFIM'

    with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
        with wf.graph.as_default():
            Eloc=tf.placeholder(dtype=tf.float64,shape=[numsamples])
            samp=tf.placeholder(dtype=tf.int32,shape=[numsamples,N])
            log_probs_=wf.log_probability(samp,inputdim=2)

            #now calculate the fake cost function to enjoy the properties of automatic differentiation
            cost = tf.reduce_mean(tf.multiply(log_probs_,Eloc)) - tf.reduce_mean(Eloc)*tf.reduce_mean(log_probs_)

            #Calculate Gradients---------------

            gradients, variables = zip(*optimizer.compute_gradients(cost))

            #End calculate Gradients---------------

            optstep=optimizer.apply_gradients(zip(gradients,variables), global_step = global_step)
            sess.run(tf.variables_initializer(optimizer.variables()))
            saver=tf.train.Saver(max_to_keep=50)
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
                meanEnergy=np.load(save_dir + '/meanEnergy_N'+str(N)+'_samp'+str(numsamples)+'_Jz'+str(Jz[0])+'_Bx'+str(Bx)+'_GRURNN_OBC'+ savename + ending + '.npy').tolist()
                varEnergy=np.load(save_dir + '/varEnergy_N'+str(N)+'_samp'+str(numsamples)+'_Jz'+str(Jz[0])+'_Bx'+str(Bx)+'_GRURNN_OBC'+ savename + ending + '.npy').tolist()
        
        #Loading previous random states
        py_random_state, np_random_state, tf_random_state = open 
    #------------------------------------

    with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
        with wf.graph.as_default():


          samples_ = wf.sample(numsamples=numsamples,inputdim=2)
          samples = np.ones((numsamples, N), dtype=np.int32)

          samples_placeholder=tf.placeholder(dtype=tf.int32,shape=(None,N))
          log_probs_tensor=wf.log_probability(samples_placeholder,inputdim=2)

          queue_samples = np.zeros((N+1, numsamples, N), dtype = np.int32) #Array to store all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)
          log_probs = np.zeros((N+1)*numsamples, dtype=np.float64) #Array to store the log_probs of all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)


          for it in range(len(meanEnergy),numsteps+1):

              samples=sess.run(samples_)
                
              #Estimating local_energies
              local_energies = Ising_local_energies(Jz, Bx, samples, queue_samples, log_probs_tensor, samples_placeholder, log_probs, batch_size, sess)

              meanE = np.mean(local_energies)
              varE = np.var(local_energies)

              #adding elements to be saved
              meanEnergy.append(meanE)
              varEnergy.append(varE)

              sess.run(optstep,feed_dict={Eloc:local_energies,samp:samples,learningrate_placeholder: lr})

             #Comment if you don't want to save
              if it%checkpoint_steps==0:
                    print("Step: ", it)
                    #Saving the performances
                    np.save(save_dir + '/meanEnergy_N'+str(N)+'_samp'+str(numsamples)+'_Jz'+str(Jz[0])+'_Bx'+str(Bx)+'_GRURNN_OBC'+ savename + ending + '.npy',meanEnergy)
                    np.save(save_dir + '/varEnergy_N'+str(N)+'_samp'+str(numsamples)+'_Jz'+str(Jz[0])+'_Bx'+str(Bx)+'_GRURNN_OBC'+ savename + ending + '.npy',varEnergy)
                    #Saving the model 
                    saver.save(sess,path+'/'+filename, global_step = it)
        
        if (numsteps%checkpoint_steps!=0):
            #Saving the performances
                np.save(save_dir + '/meanEnergy_N'+str(N)+'_samp'+str(numsamples)+'_Jz'+str(Jz[0])+'_Bx'+str(Bx)+'_GRURNN_OBC'+ savename + ending + '.npy',meanEnergy)
                np.save(save_dir + '/varEnergy_N'+str(N)+'_samp'+str(numsamples)+'_Jz'+str(Jz[0])+'_Bx'+str(Bx)+'_GRURNN_OBC'+ savename + ending + '.npy',varEnergy)
            #Saving the model 
                saver.save(sess,path+'/'+filename, global_step = it)
                
    return meanEnergy, varEnergy
    #----------------------------------------------------------------------------------------------



def compute_correlations_from_model(numsamples = 10**6, old_numsamples = 500, 
                                    systemsize = 20, num_units = 50, Bx = 1, num_layers = 1, 
                                    batch_size = 25000, seed = 111, 
                                    symRNN = 'None', entropy = False,
                                    save_dir = "../Check_Points/1DTFIM", model_step = None, max_samp = np.inf):

    #Seeding ---------------------------------------------
    tf.reset_default_graph()
    random.seed(seed)  # `python` built-in pseudo-random generator
    np.random.seed(seed)  # numpy pseudo-random generator
    tf.set_random_seed(seed)  # tensorflow pseudo-random generator
    #End Seeding ---------------------------------------------
    
    # System size
    N = systemsize
    Jz = +np.ones(N) #Ferromagnetic coupling

    # Intitializing the RNN-----------
    units=[num_units]*num_layers #list containing the number of hidden units for each layer of the networks


    if (symRNN in ['None', 'none']): 
        wf=RNNwavefunction(N,units=units,cell=tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell, seed = seed) #contains the graph with the RNNs
    elif (symRNN == 'parity'):
        wf=SymRNNwavefunction(N,units=units,cell=tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell, seed = seed)
    elif (symRNN == 'spinflip'):
        wf=SpinFlipSymRNNwavefunction(N,units=units,cell=tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell, seed = seed)
    else:
        print("Unknown symmetry: ", symRNN)
        print("Accepted values are: 'None'/'none', 'parity' or 'spinflip'")
        return -1, -1, -1, -1, -1, -1, -1
       
    numsamples_ = 20
    sampling=wf.sample(numsamples_,inputdim = 2) #call this function once to create the dense layers

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

    steps_string = ''
    if model_step != None:
        steps_string = '-{}'.format(model_step)
    
    savename = '_TFIM'
    filename= save_dir + '/RNNwavefunction_N'+str(N)+'_samp'+str(old_numsamples)+'_Jz1Bx'+str(Bx)+'_GRURNN_OBC'+ending + '.ckpt' + steps_string
    #-----------------------------
    #Restoring old session
    #and computing quantities
    

    with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
        with wf.graph.as_default():
            
            saver=tf.train.Saver()
            saver.restore(sess,path+'/'+filename)         
            original_samp = numsamples

            if (numsamples>max_samp):
                steps = ceil (numsamples/max_samp)
                new_samp = numsamples//steps
                middle_spin = ceil(N/2)
                z_sum = np.zeros( N - middle_spin, dtype = np.float64)
                x_sum = np.zeros( N - middle_spin, dtype = np.float64)
                z_sum2 = np.zeros(N - middle_spin, dtype = np.float64)
                x_sum2 = np.zeros(N - middle_spin, dtype = np.float64)
                sumE = 0
                sumE2 = 0
                swap_sum = 0
                swap_sum2 = 0
                print(steps)
                for i in range(steps):
                    print("Step: {}".format(i))
                    if i < steps-1:
                        numsamples = new_samp
                    else:
                        numsamples = original_samp - (steps -1)*new_samp

                    samples_ = wf.sample(numsamples=numsamples,inputdim=2)

                    samples_placeholder=tf.placeholder(dtype=tf.int32,shape=(None,N))
                    log_probs_tensor=wf.log_probability(samples_placeholder,inputdim=2)

                    queue_samples = np.zeros((N+1, numsamples, N), dtype = np.int32) #Array to store all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)
                    log_probs = np.zeros((N+1)*numsamples, dtype=np.float64) #Array to store the log_probs of all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)

                    samples=sess.run(samples_)
                    samples_B = sess.run(samples_)

                    #Estimating local_energies
                    local_energies = Ising_local_energies(Jz, Bx, samples, queue_samples, log_probs_tensor, samples_placeholder, log_probs, batch_size, sess)
                    sumE += np.sum(local_energies)
                    sumE2 += np.dot(local_energies, local_energies)
                    z = Ising_SzSz_correlations(samples)
                    x = Ising_SxSx_correlations(samples, queue_samples, log_probs_tensor, samples_placeholder, log_probs, batch_size, sess)
                    z_sum += np.sum(z, axis=1)
                    z_sum2 += np.sum(z*z, axis=1)
                    x_sum += np.sum(x, axis=1)
                    x_sum2 += np.sum(x*x, axis=1)

                    if entropy:
                        swap_values = Swap_operator(samples, samples_B, np.arange(0, N/2, dtype=np.int32), log_probs_tensor, samples_placeholder, batch_size, sess)
                        swap_sum += np.sum(swap_values)
                        #swap_sum2 += np.dot(swap_values, swap_values)

                meanE = sumE/original_samp
                varE = sumE2/original_samp - meanE*meanE
                z_correlations = z_sum /original_samp
                x_correlations = x_sum /original_samp
                z_corr_var = z_sum2 /original_samp - z_correlations*z_correlations #element wise operation
                x_corr_var = x_sum2 /original_samp - x_correlations*x_correlations #element wise operation

                r_ent = Renyi_Entropy(swap_sum/original_samp) if entropy else -1
            else:
                samples_ = wf.sample(numsamples=numsamples,inputdim=2)
            
                samples_placeholder=tf.placeholder(dtype=tf.int32,shape=(None,N))
                log_probs_tensor=wf.log_probability(samples_placeholder,inputdim=2)
                queue_samples = np.zeros((N+1, numsamples, N), dtype = np.int32) #Array to store all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)
                log_probs = np.zeros((N+1)*numsamples, dtype=np.float64) #Array to store the log_probs of all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)
                samples=sess.run(samples_)
                #Estimating local_energies
                local_energies = Ising_local_energies(Jz, Bx, samples, queue_samples, log_probs_tensor, samples_placeholder, log_probs, batch_size, sess)
                meanE = np.mean(local_energies)
                varE = np.var(local_energies)
                z = Ising_SzSz_correlations(samples)
                x  = Ising_SxSx_correlations(samples, queue_samples, log_probs_tensor, samples_placeholder, log_probs, batch_size, sess)
                z_correlations = np.mean(z, axis=1)
                x_correlations = np.mean(x, axis=1)
                z_corr_var = np.var(z, axis=1)
                x_corr_var = np.var(x, axis=1)
            
                samples_B = sess.run(samples_)
                r_ent = Renyi_Entropy(np.mean( Swap_operator(samples, samples_B, np.arange(0, N/2, dtype=np.int32), log_probs_tensor, samples_placeholder, batch_size, sess))) if entropy else -1

            np.save(save_dir + '/SzSzCorrelation_N'+str(N)+'_samp'+str(original_samp)+'_Jz'+str(Jz[0])+'_Bx'+str(Bx)+'_GRURNN_OBC'+ savename + ending + steps_string + '.npy',z_correlations)
            np.save(save_dir + '/SxSxCorrelation_N'+str(N)+'_samp'+str(original_samp)+'_Jz'+str(Jz[0])+'_Bx'+str(Bx)+'_GRURNN_OBC'+ savename + ending + steps_string + '.npy',x_correlations)

            np.save(save_dir + '/SzSzCorrelationVariance_N'+str(N)+'_samp'+str(original_samp)+'_Jz'+str(Jz[0])+'_Bx'+str(Bx)+'_GRURNN_OBC'+ savename + ending + steps_string + '.npy',z_corr_var)
            np.save(save_dir + '/SxSxCorrelationVariance_N'+str(N)+'_samp'+str(original_samp)+'_Jz'+str(Jz[0])+'_Bx'+str(Bx)+'_GRURNN_OBC'+ savename + ending + steps_string + '.npy',x_corr_var)
            
            np.save(save_dir + '/mean_var_renyiE_N'+str(N)+'_samp'+str(original_samp)+'_Jz'+str(Jz[0])+'_Bx'+str(Bx)+'_GRURNN_OBC'+ savename + ending + steps_string + '.npy', (meanE, varE , r_ent)) 
            
    return z_correlations, z_corr_var, x_correlations, x_corr_var, meanE, varE, r_ent


def Swap_operator (samples_1, samples_2, indices_A, log_probs_tensor, samples_placeholder, batch_size, sess):
    """
    Computing the average value of Swap Operator, as a help for Renyi 2 Entropy
    --------------------------------------------------------------------------------------------------------
    Parameters:
    - samples_A: (numsamples, N) --> "real" system
    - samples_B: (numsamples, N) --> "ancillary" system
    - indices_A: (l,)            --> indices of the spins in one of the 2 subsystem (the one not traced out)
    - log_probs_tensor:          --> A TF tensor with size (None)
    - samples_placeholder:       --> A TF placeholder to feed in a set of configurations
    - sess:                      --> The current TF session
    --------------------------------------------------------------------------------------------------------
    Returns: float64, the computed Reny Entropy
    """
    if samples_1.shape != samples_2.shape:
        return -1
    
    numsamples = samples_1.shape[0]
    N = samples_1.shape[1]

    swapped_samples_1 = np.copy(samples_1)
    swapped_samples_2 = np.copy(samples_2)

    #swapping the subsystem samples:
    for i in indices_A:
        swapped_samples_1[:,i] = samples_2[:, i]
        swapped_samples_2[:,i] = samples_1[:, i]

    log_probs_1 = np.zeros(numsamples, dtype=np.float64)
    log_probs_2 = np.zeros(numsamples, dtype=np.float64) 
    log_probs_swapped_1 = np.zeros(numsamples, dtype=np.float64) 
    log_probs_swapped_2 = np.zeros(numsamples, dtype=np.float64) 

    #Calculating log_probs from samples in steps
    steps = ceil(numsamples/batch_size) #Get a maximum of 25000 configurations in batch size just to not allocate too much memory

    for i in range(steps):
        if i < steps-1:
            cut = slice((i*numsamples)//steps,((i+1)*numsamples)//steps)      #in this way batches are equally balanced (if 48k tot config - > 24k, 24k, not 25, 23)
        else:
            cut = slice((i*numsamples)//steps,numsamples)
        log_probs_1[cut] = sess.run(log_probs_tensor, feed_dict={samples_placeholder:samples_1[cut]})
        log_probs_2[cut] = sess.run(log_probs_tensor, feed_dict={samples_placeholder:samples_2[cut]})
        log_probs_swapped_1[cut] = sess.run(log_probs_tensor, feed_dict={samples_placeholder:swapped_samples_1[cut]})
        log_probs_swapped_2[cut] = sess.run(log_probs_tensor, feed_dict={samples_placeholder:swapped_samples_2[cut]})

    
    Swap_operator =  np.exp(0.5*(log_probs_swapped_1 + log_probs_swapped_2 - log_probs_1 - log_probs_2)) 
    return Swap_operator

def Renyi_Entropy(swap):
    return -np.log(swap)

def Sz_magnetization (samples):
    """
    Computing the magnetization along axis z
    --------------------------------------------------------------------------------------------------------
    Parameters:
    - samples: (numsamples, N)
    --------------------------------------------------------------------------------------------------------
    Returns: ndarray of size (numsamples, N) that are samples evaluated as one-half spins
    """
    N = samples.shape[1]
    return samples-0.5

def Sx_magnetization (samples, queue_samples, log_probs_tensor, samples_placeholder, log_probs, batch_size, sess):
    """
    Computing the magnetization along axis x
    --------------------------------------------------------------------------------------------------------
    Parameters:
    - samples: (numsamples, N)
    - queue_samples: ((N+1)*numsamples, N) an empty allocated np array to store the non diagonal elements
    - log_probs_tensor: A TF tensor with size (None)
    - samples_placeholder: A TF placeholder to feed in a set of configurations
    - log_probs: ((N+1)*numsamples) an empty allocated np array to store the log_probs non diagonal elements
    - sess: The current TF session
    --------------------------------------------------------------------------------------------------------
    Returns: ndarray of size (N,) with the sum over the samples of Sx_i magnetization in i-th position
    """
    numsamples = samples.shape[0]
    N = samples.shape[1]
    
    #store the diagonal samples -> needed for the evaluation of matrix element
    queue_samples[0] = samples 
    
    for i in range(N):  #Non-diagonal elements
        valuesT = np.copy(samples)
        valuesT[:,i][samples[:,i]==1] = 0 #Flip to 0 spin i of each sampling, if in the original was 1
        valuesT[:,i][samples[:,i]==0] = 1 #Flip to 1 spin i of each sampling, if in the original was 0
        queue_samples[i+1] = valuesT

    #Calculating log_probs from samples in steps
    len_sigmas = (N+1)*numsamples
    steps = ceil(len_sigmas/batch_size) #Get a maximum of 25000 configurations in batch size just to not allocate too much memory

    queue_samples_reshaped = np.reshape(queue_samples, [(N+1)*numsamples, N]) #cause it was ((N+1), numsamples, N)

    for i in range(steps):
        if i < steps-1:
            cut = slice((i*len_sigmas)//steps,((i+1)*len_sigmas)//steps)      #in this way batches are equally balanced (if 48k tot config - > 24k, 24k, not 25, 23)
        else:
            cut = slice((i*len_sigmas)//steps,len_sigmas)
        log_probs[cut] = sess.run(log_probs_tensor, feed_dict={samples_placeholder:queue_samples_reshaped[cut]})

    log_probs_reshaped = np.reshape(log_probs, [N+1,numsamples]) #it was ( (N+1)*numsamples )
    
    Sx_mag = 0.5 * np.exp(0.5*log_probs_reshaped[1:,:]-0.5*log_probs_reshaped[0,:])
    return Sx_mag



def compute_mag_from_model(numsamples = 10**6, old_numsamples = 500, 
                                    systemsize = 20, num_units = 50, Bx = 1, num_layers = 1, 
                                    batch_size = 25000, seed = 111, 
                                    symRNN = False,
                                    save_dir = "../Check_Points/1DTFIM", model_step = None, max_samp = np.inf):

    #Seeding ---------------------------------------------
    tf.reset_default_graph()
    random.seed(seed)  # `python` built-in pseudo-random generator
    np.random.seed(seed)  # numpy pseudo-random generator
    tf.set_random_seed(seed)  # tensorflow pseudo-random generator
    #End Seeding ---------------------------------------------
    
    # System size
    N = systemsize
    Jz = +np.ones(N) #Ferromagnetic coupling

    # Intitializing the RNN-----------
    units=[num_units]*num_layers #list containing the number of hidden units for each layer of the networks


    if (symRNN): 
        wf=SpinFlipSymRNNwavefunction(N,units=units,cell=tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell, seed = seed)
    else:
        wf=RNNwavefunction(N,units=units,cell=tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell, seed = seed) #contains the graph with the RNNs
    
    numsamples_ = 20
    sampling=wf.sample(numsamples_,inputdim = 2) #call this function once to create the dense layers

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
    
    if symRNN:
        ending += '_symm'
    steps_string = ''
    if model_step != None:
        steps_string = '-{}'.format(model_step)
    
    savename = '_TFIM'
    filename= save_dir + '/RNNwavefunction_N'+str(N)+'_samp'+str(old_numsamples)+'_Jz1Bx'+str(Bx)+'_GRURNN_OBC'+ending + '.ckpt' + steps_string
    #-----------------------------
    #Restoring old session
    #and computing quantities
    

    with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
        with wf.graph.as_default():
            
            saver=tf.train.Saver()
            saver.restore(sess,path+'/'+filename)         
            original_samp = numsamples

            if (numsamples>max_samp):
                steps = ceil (numsamples/max_samp)
                new_samp = numsamples//steps
                z_sum = np.zeros(N, dtype=np.float32)
                x_sum = np.zeros(N, dtype=np.float32)
                z_sum2 = np.zeros(N, dtype=np.float32)
                x_sum2 = np.zeros(N, dtype=np.float32)
                print(steps)
                for i in range(steps):
                    print("Step: {}".format(i))
                    if i < steps-1:
                        numsamples = new_samp
                    else:
                        numsamples = original_samp - (steps -1)*new_samp

                    samples_ = wf.sample(numsamples=numsamples,inputdim=2)

                    samples_placeholder=tf.placeholder(dtype=tf.int32,shape=(None,N))
                    log_probs_tensor=wf.log_probability(samples_placeholder,inputdim=2)

                    queue_samples = np.zeros((N+1, numsamples, N), dtype = np.int32) #Array to store all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)
                    log_probs = np.zeros((N+1)*numsamples, dtype=np.float64) #Array to store the log_probs of all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)

                    samples=sess.run(samples_)

                    #Estimating local_energies
                    z = Sz_magnetization(samples)
                    x = Sx_magnetization(samples, queue_samples, log_probs_tensor, samples_placeholder, log_probs, batch_size, sess)
                    z_sum += np.sum(z, axis = 0)
                    z_sum2 += np.sum(z*z, axis=0)
                    x_sum += np.sum(x, axis=1)
                    x_sum2 += np.sum(x*x, axis=1)

                z_mag = z_sum /original_samp
                x_mag = x_sum /original_samp
                z_mag_var = z_sum2 /original_samp - z_mag*z_mag #element wise operation
                x_mag_var = x_sum2 /original_samp - x_mag*x_mag #element wise operation
            else:
                samples_ = wf.sample(numsamples=numsamples,inputdim=2)
            
                samples_placeholder=tf.placeholder(dtype=tf.int32,shape=(None,N))
                log_probs_tensor=wf.log_probability(samples_placeholder,inputdim=2)
                queue_samples = np.zeros((N+1, numsamples, N), dtype = np.int32) #Array to store all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)
                log_probs = np.zeros((N+1)*numsamples, dtype=np.float64) #Array to store the log_probs of all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)
                samples=sess.run(samples_)

                z = Sz_magnetization(samples)
                x  = Sx_magnetization(samples, queue_samples, log_probs_tensor, samples_placeholder, log_probs, batch_size, sess)
                z_mag = np.mean(z, axis=0)
                x_mag = np.mean(x, axis=1)
                z_mag_var = np.var(z, axis=0)
                x_mag_var = np.var(x, axis=1)
                     
            np.save(save_dir + '/mean_var_mag_z_x_N'+str(N)+'_samp'+str(original_samp)+'_Jz'+str(Jz[0])+'_Bx'+str(Bx)+'_GRURNN_OBC'+ savename + ending + steps_string + '.npy', (z_mag, z_mag_var, x_mag, x_mag_var)) 
            
    return z_mag, z_mag_var, x_mag, x_mag_var



def sample_from_model(numsamples = 10**6, old_numsamples = 500, 
                                    systemsize = 20, Bx = 1, num_units = 50, num_layers = 1, 
                                    seed = 111, symRNN = False,
                                    save_dir = "../Check_Points/1DTFIM", model_step = None):

    #Seeding ---------------------------------------------
    tf.reset_default_graph()
    random.seed(seed)  # `python` built-in pseudo-random generator
    np.random.seed(seed)  # numpy pseudo-random generator
    tf.set_random_seed(seed)  # tensorflow pseudo-random generator
    #End Seeding ---------------------------------------------
    
    # System size
    N = systemsize
    Jz = +np.ones(N) #Ferromagnetic coupling

    # Intitializing the RNN-----------
    units=[num_units]*num_layers #list containing the number of hidden units for each layer of the networks


    if (symRNN): 
        wf=SpinFlipSymRNNwavefunction(N,units=units,cell=tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell, seed = seed)
    else:
        wf=RNNwavefunction(N,units=units,cell=tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell, seed = seed) #contains the graph with the RNNs
    
    numsamples_ = 20
    sampling=wf.sample(numsamples_,inputdim = 2) #call this function once to create the dense layers

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
    
    if symRNN:
        ending += '_symm'
    steps_string = ''
    if model_step != None:
        steps_string = '-{}'.format(model_step)
    
    savename = '_TFIM'
    filename= save_dir + '/RNNwavefunction_N'+str(N)+'_samp'+str(old_numsamples)+'_Jz1Bx'+str(Bx)+'_GRURNN_OBC'+ending + '.ckpt' + steps_string
    #-----------------------------
    #Restoring old session
    #and computing quantities
    

    with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
        with wf.graph.as_default():
            saver=tf.train.Saver()
            saver.restore(sess,path+'/'+filename)

            samples_ = wf.sample(numsamples=numsamples,inputdim=2)
            samples=sess.run(samples_)

            np.save(save_dir + '/samples_N'+str(N)+'_samp'+str(numsamples)+'_Jz'+str(Jz[0])+'_Bx'+str(Bx)+'_GRURNN_OBC'+ savename + ending + steps_string + '.npy', samples) 
            
    return samples


def get_variable(  var_name, 
                model_numsamples = 500, systemsize = 20, Bx = 1, 
                num_units = 50, num_layers = 1, 
                seed = 111, symRNN = False,
                save_dir = "../Check_Points/1DTFIM", model_step = None):
    #Seeding ---------------------------------------------
    tf.reset_default_graph()
    random.seed(seed)  # `python` built-in pseudo-random generator
    np.random.seed(seed)  # numpy pseudo-random generator
    tf.set_random_seed(seed)  # tensorflow pseudo-random generator
    #End Seeding ---------------------------------------------
    
    # System size
    N = systemsize
    Jz = +np.ones(N) #Ferromagnetic coupling

    # Intitializing the RNN-----------
    units=[num_units]*num_layers #list containing the number of hidden units for each layer of the networks


    if (symRNN): 
        wf=SpinFlipSymRNNwavefunction(N,units=units,cell=tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell, seed = seed)
    else:
        wf=RNNwavefunction(N,units=units,cell=tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell, seed = seed) #contains the graph with the RNNs
    
    numsamples_ = 20
    sampling=wf.sample(numsamples_,inputdim = 2) #call this function once to create the dense layers

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
    
    if symRNN:
        ending += '_symm'
    steps_string = ''
    if model_step != None:
        steps_string = '-{}'.format(model_step)
    
    savename = '_TFIM'
    filename= save_dir + '/RNNwavefunction_N'+str(N)+'_samp'+str(model_numsamples)+'_Jz1Bx'+str(Bx)+'_GRURNN_OBC'+ending + '.ckpt' + steps_string
    #-----------------------------
    #Restoring old session
    #and computing quantities
    

    with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
        with wf.graph.as_default():
            saver=tf.train.Saver()
            saver.restore(sess,path+'/'+filename)
    #Selecting the variable
            var = [v for v in tf.trainable_variables() if v.name==var_name][0] #[0] is to get the value ([1] is the type)
            val = sess.run(var)
    return val 