from operator import setitem
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #stop displaying tensorflow warnings
import numpy as np
import os
import random
from math import ceil

from ComplexRNNwavefunction import RNNwavefunction
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


    #Sum y-interactions (even indices) with the correct sign
    local_energies += J2*np.sum(np.array(y_states_sign_map) * np.exp(log_ampl_reshaped[2::2,:]-log_ampl_reshaped[0,:]), axis = 0)

    return local_energies
#--------------------------

#def Entropy():












# ---------------- Running VMC with RNNs -------------------------------------
def run_Kita1D(numsteps = 10**4, systemsize = 20, num_units = 50, J1_ = 1, J2_ = 1, J3_=1, 
                num_layers = 1, numsamples = 200, learningrate = 5e-3, 
                annealing_parameters = None, annealing_entropy = None,
                batch_size = 25000, seed = 111, 
                load_model = False, 
                save_dir = ".", checkpoint_steps = 11000):

    """
    annealing_parameters = (j1, j2, j3, steps) tuple with the starting parameters for annealing and the decaying steps
    """
    #Seeding ---------------------------------------------
    tf.reset_default_graph()
#   random.seed(seed)  # `python` built-in pseudo-random generator
#    np.random.seed(seed)  # numpy pseudo-random generator
#    tf.set_random_seed(seed)  # tensorflow pseudo-random generator

    #End Seeding ---------------------------------------------

    # System size
    N = systemsize

    #Learning rate
    lr=np.float64(learningrate)

    # Intitializing the RNN-----------
    units=[num_units]*num_layers #list containing the number of hidden units for each layer of the networks

    input_dim=4 #Dimension of the Hilbert space for each site (here = 4, up or down)
    numsamples_=20 #only for initialization; later I'll use a much larger value (see below)

    

    wf=RNNwavefunction(N, inputdim=4, units=units,cell=tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell, seed = seed) #contains the graph with the RNNs
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
    
    param_string = 'N{N}_samp{samp}_J1{j1:.1f}_J2{j2:.1f}_J3{j3:.1f}_GRURNN_OBC'.format(N=N, samp=numsamples, j1=J1_, j2=J2_, j3=J3_)
    
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

        ### Set Annealing process for parameters
            if annealing_parameters!=None:
                    J1, J2, J3, param_steps = annealing_parameters
                    tot_steps = numsteps-len(meanEnergy) + 1
                    decay_steps = tot_steps/param_steps
                    j1_delta = (J1 - J1_) /decay_steps
                    j2_delta = (J2 - J2_) /decay_steps
                    j3_delta = (J3 - J3_ )/decay_steps
                    print("Performing annealing...")
                    print("Starting with j1 = {j1}, j2 = {j2}, j3 = {j3}".format(j1=J1, j2=J2, j3=J3))
                    print("Changing parameters every {s} steps: j1-={j1}, j2-={j2}, j3-={j3}".format(s=param_steps, j1=j1_delta, j2=j2_delta, j3=j3_delta))
            else:
                param_steps = numsteps + 100
                J1, J2, J3 = (J1_, J2_, J3_)
                j1_delta = 0.0
                j2_delta = 0.0
                j3_delta = 0.0

            if annealing_entropy!=None:
                lambda_, entropy_steps = annealing_entropy
            else:
                lambda_, entropy_steps = (0.0, numsteps + 100)

            starting_steps = len(meanEnergy)
            for it in range(starting_steps,numsteps+1):

                samples=sess.run(samples_)

                if (it - starting_steps)%param_steps == 0:
                    J1 -= j1_delta
                    J2 -= j2_delta
                    J3 -= j3_delta
                    print("Step: ", it, " j1-->{j1}, j2-->{j2}, j3-->{j3}".format(j1=J1, j2=J2, j3=J3))
                #Estimating local_energies
                local_energies = Kitaev_local_energies(J1, J2, J3, samples, queue_samples, log_ampl_tensor, samples_placeholder, log_amplitudes, batch_size, sess)
                #if (it - starting_steps)%entropy_steps == 0:
                    #local_energies += lambda_ * Entropy()
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
                      seed = 111,
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



    wf=RNNwavefunction(N, inputdim=4, units=units,cell=tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell, seed = seed) #contains the graph with the RNNs
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