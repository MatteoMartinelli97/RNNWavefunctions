import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #stop displaying tensorflow warnings
import numpy as np
import os
import time
import random
from math import ceil

from Complex_RNNwavefunction import RNNwavefunction
from MDRNNcell import MDRNNcell

# Loading Functions --------------------------
def Kitaev2D_local_energies(J1, J2, J3,  Nx, Ny, samples, queue_samples, log_ampl_tensor, samples_placeholder, log_amplitudes, entropy_lambda, batch_size, sess):
    """ To get the local energies of 2D Kitaev Model (OBC) given a set of set of samples in parallel!
    Returns: The local energies that correspond to the "samples"
    Inputs:
    - samples: (numsamples, Ny,Nx)
    - J1, J2, J3: float
    - queue_samples: ((Nx*Ny+1)*numsamples, Nx,Ny) an empty allocated np array to store the non diagonal elements
    - log_probs_tensor: A TF tensor with size (None)
    - samples_placeholder: A TF placeholder to feed in a set of configurations
    - log_probs: ((Nx*Ny+1)*numsamples): an empty allocated np array to store the log_probs non diagonal elements
    - sess: The current TF session
    """

    numsamples = samples.shape[0]

    N = Nx*Ny #Total number of spins

    local_energies = np.zeros((numsamples), dtype = np.complex128)

    for i in range(0, Nx, 2): 
        #diagonal elements even for intra-rung
        valuesT = np.copy(samples[:,:,i])
        valuesT[samples[:,:,i]==1] = -1 #Opposite spin
        valuesT[samples[:,:,i]==2] = -1 #Opposite spin
        valuesT[samples[:,:,i]==3] = +1 #Same spin
        valuesT[samples[:,:,i]==0] = +1 #Same spin
        #sum over y-axis = sum over all the rows
        local_energies += np.sum(valuesT*J3, axis=1)
    
    for i in range(Ny): 
        #diagonal elements odd for inter-rung
        #first and last row are linked by PBC 
        valuesA = np.copy(samples[:, i-1, 1::2])

        #Map to actual spins and compute the interaction on z-axis
        #Upper row, take the lower spin
        valuesA[samples[:, i-1, 1::2]==0] = -1
        valuesA[samples[:, i-1, 1::2]==2] = -1
        valuesA[samples[:, i-1, 1::2]==1] = +1
        valuesA[samples[:, i-1, 1::2]==3] = +1

        valuesB = np.copy(samples[:, i, 1::2])
        #Lower row, take the upper spin
        valuesB[samples[:, i, 1::2]==0] = -1
        valuesB[samples[:, i, 1::2]==1] = -1
        valuesB[samples[:, i, 1::2]==2] = +1
        valuesB[samples[:, i, 1::2]==3] = +1

        local_energies += np.sum(J3 * valuesA * valuesB, axis=1)
        
    queue_samples[0] = samples #storing the diagonal samples

    #Non-diagonal elements are the same as 1D chain
    y_states_sign_map = []
    Ns = 2*Nx -2 #Number of new samples per row
    if J1 != 0 or J2 !=0:
        #Loop over rows
        for i in range(Ny):
            for j in range(Nx-1):
                #Even   #SxSx
                        #SySy
                if (j%2==0):
                    #off-diag interactions are always on the same row i
                    
                    #Flip upper spin j and j+1 --> upper spin flip = +2 with 4 = 0
                    x_interaction_states = np.copy(samples)
                    x_interaction_states[:,i, j] = (2 + x_interaction_states[:,i, j]) % 4
                    x_interaction_states[:,i, j+1] = (2 + x_interaction_states[:,i, j+1]) % 4

                    #Always remember x-interactions in odd index
                    queue_samples[2*j+1 + Ns * i] = x_interaction_states 
                    
                    #Flip lower spin j and j+1 --> lower spin flip = +1 if even, else -1
                    # 1 - 2* ([2k, 2k+1] % 2) --> 1 - 2*([0,1]) --> [1, -1]
                    y_interaction_states = np.copy(samples)
                    y_interaction_states[:,i, j] += 1 - 2 * (y_interaction_states[:,i, j] % 2)
                    y_interaction_states[:,i, j+1] += 1 - 2 * (y_interaction_states[:,i, j+1] % 2)

                    #For lower spin y interaction is negative if the sum of the 2 new states is even
                    #Remember the sign of each of these states
                    # 1 - 2*[T,F] --> [-1, +1]
                    sign_map = 1 - 2 * ( (y_interaction_states[:, i, j] + y_interaction_states[:, i, j+1])%2 == 0 )
                    y_states_sign_map.append(sign_map)

                    #Always remember y-interactions in even index
                    queue_samples[2*j+2 + Ns * i] = y_interaction_states


                #Odd    #SySy
                        #SxSx
                else:

                    #Flip upper spin j and j+1 --> upper spin flip = +2 with 4 = 0
                    y_interaction_states = np.copy(samples)
                    y_interaction_states[:,i, j] = (2 + y_interaction_states[:,i, j]) % 4
                    y_interaction_states[:,i, j+1] = (2 + y_interaction_states[:,i, j+1]) % 4
                    
                    #For upper spin y interaction is negative if the 2 new states are the same, or if their sum%4 is 1 (0,1/2,3)
                    #Remember the sign of each of these states
                    # 1 - 2*[T,F] --> [-1, +1]
                    sign_map = 1 - 2 * ( (y_interaction_states[:, i, j] == y_interaction_states[:, i, j+1]) |  ((y_interaction_states[:, i, j] + y_interaction_states[:, i, j+1])%4 == 1) )
                    y_states_sign_map.append(sign_map)

                    #Always remember y-interactions in even index
                    queue_samples[2*j+2 + Ns * i] = y_interaction_states
                    
                    #Flip lower spin i and i+1 --> lower spin flip = +1 if even, else -1
                    # 1 - 2* ([2k, 2k+1] % 2) --> 1 - 2*([0,1]) --> [1, -1]
                    x_interaction_states = np.copy(samples)
                    x_interaction_states[:,i, j] += 1 - 2 * (x_interaction_states[:,i, j] % 2)
                    x_interaction_states[:,i, j+1] += 1 - 2 * (x_interaction_states[:,i, j+1] % 2)

                    #Always remember x-interactions in odd index
                    queue_samples[2*j+1 + Ns * i] = x_interaction_states

    #Calculating log_probs from samples
    #Do it in steps

    len_sigmas = (Ny*Ns+1)*numsamples
    steps = ceil(len_sigmas/batch_size) #Get a maximum of 25000 configurations in batch size to not allocate too much memory

    queue_samples_reshaped = np.reshape(queue_samples, [(Ny*Ns+1)*numsamples, Ny,Nx]) #from  ( (Ny*Ns+1), numsamples, Ny, Nx )
    for i in range(steps):
      if i < steps-1:
          cut = slice((i*len_sigmas)//steps,((i+1)*len_sigmas)//steps)
      else:
          cut = slice((i*len_sigmas)//steps,len_sigmas)
      log_amplitudes[cut] = sess.run(log_ampl_tensor, feed_dict={samples_placeholder:queue_samples_reshaped[cut]})

    log_ampl_reshaped = np.reshape(log_amplitudes, [Ny*Ns+1,numsamples])
    #Sum x interactions (odd indices)
    local_energies += J1*np.sum(np.exp(log_ampl_reshaped[1::2,:]-log_ampl_reshaped[0,:]), axis = 0)

    #Sum y-interactions (even indices) with the correct sign
    local_energies += J2*np.sum(np.array(y_states_sign_map) * np.exp(log_ampl_reshaped[2::2,:]-log_ampl_reshaped[0,:]), axis = 0)

    return local_energies + entropy_lambda*np.real(log_ampl_reshaped[0, :])
#--------------------------


# ---------------- Running VMC with 2DRNNs -------------------------------------
def run_2DKitaev(numsteps = 2*10**4, systemsize_x = 5, systemsize_y = 5, 
                 J1_ = 1, J2_ = 1, J3_=1, 
                 num_units = 50, numsamples = 500, learningrate = 5e-3, 
                 entropy_params = None,
                 batch_size=50000, seed = 111,
                 load_model = False, model_step=None,
                 save_dir = ".", checkpoint_steps = 11000):

    #Seeding
    tf.compat.v1.reset_default_graph()
    random.seed(seed)  # `python` built-in pseudo-random generator
    np.random.seed(seed)  # numpy pseudo-random generator
    tf.compat.v1.set_random_seed(seed)  # tensorflow pseudo-random generator
    tf.compat.v1.disable_v2_behavior()

    entropy = entropy_params != None

    # Intitializing the RNN-----------
    units=[num_units] #list containing the number of hidden units for each layer of the networks (We only support one layer for the moment)

    Nx=systemsize_x #x dim
    Ny=systemsize_y #y dim

    lr=np.float64(learningrate)



    input_dim=4 #Dimension of the Hilbert space for each site (here = 4, up or down)
    numsamples_=20 #number of samples only for initialization
    wf=RNNwavefunction(Nx,Ny,hilbert_dim= 4, units=units,cell=MDRNNcell,seed = seed) #contains the graph with the RNNs

    sampling=wf.sample(numsamples_,input_dim) #call this function once to create the dense layers

    #now initialize everything --------------------
    with wf.graph.as_default():
        samples_placeholder=tf.compat.v1.placeholder(dtype=tf.int32,shape=[numsamples_,Ny,Nx]) #the samples_placeholder are the samples of all of the spins
        global_step = tf.Variable(0, trainable=False)
        learningrate_placeholder=tf.compat.v1.placeholder(dtype=tf.float64,shape=[])
        learning_rate_withexpdecay = tf.compat.v1.train.exponential_decay(learningrate_placeholder, global_step = global_step, decay_steps = 100, decay_rate = 1.0, staircase=True) #For exponential decay of the learning rate (only works if decay_rate < 1.0)
        ampl=wf.log_amplitude(samples_placeholder,input_dim) #The probs are obtained by feeding the sample of spins.
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate_withexpdecay) #Using AdamOptimizer
        init=tf.compat.v1.global_variables_initializer()
    # End Intitializing

    #Starting Session------------
    #Activating GPU
    config = tf.compat.v1.ConfigProto()
    #config.gpu_options.allow_growth = True

    sess=tf.compat.v1.Session(graph=wf.graph, config=config)
    sess.run(init)
    #---------------------------

#    with wf.graph.as_default():
#        variables_names =[v.name for v in tf.compat.v1.trainable_variables()]
#        print(variables_names)
#        sum = 0
#        values = sess.run(variables_names)
#        for k,v in zip(variables_names, values):
#            v1 = tf.reshape(v,[-1])
#            print(k,v1.shape)
#            sum +=v1.shape[0]
#        print('The sum of params is {0}'.format(sum))


    meanEnergy=[]
    varEnergy=[]

    #Running the training -------------------
    path=os.getcwd()

    print('Training with numsamples = ', numsamples)
    print('\n')

    ending='_units'
    for u in units:
        ending+='_{0}'.format(u)
    
    param_string = '{Nx}x_{Ny}y_samp{samp}_J1{j1:.1f}_J2{j2:.1f}_J3{j3:.1f}_2DVanillaRNN'.format(Nx=Nx, Ny=Ny, samp=numsamples, j1=J1_, j2=J2_, j3=J3_)

    filename= save_dir + '/RNNwavefunction_' + param_string + ending + '.ckpt'
    savename = '_2DKitaev'

    with tf.compat.v1.variable_scope(wf.scope,reuse=tf.compat.v1.AUTO_REUSE):
        with wf.graph.as_default():
            Eloc=tf.compat.v1.placeholder(dtype=tf.complex64,shape=[numsamples])
            samp=tf.compat.v1.placeholder(dtype=tf.int32,shape=[numsamples,Ny,Nx])
            log_amplitudes_=wf.log_amplitude(samp,inputdim=input_dim)

            cost = 2*tf.math.real(tf.math.reduce_mean(tf.math.conj(log_amplitudes_)*tf.stop_gradient(Eloc)) - tf.math.conj(tf.math.reduce_mean(log_amplitudes_))*tf.math.reduce_mean(tf.stop_gradient(Eloc)))
            gradients, variables = zip(*optimizer.compute_gradients(cost))

            optstep=optimizer.apply_gradients(zip(gradients,variables),global_step=global_step)
            sess.run(tf.compat.v1.variables_initializer(optimizer.variables()))

            saver=tf.compat.v1.train.Saver() #define tf saver

    #Loading previous trainings
    if load_model:
        steps_string = ''
        if model_step != None:
            steps_string = '-{}'.format(model_step)

        filename= save_dir + '/RNNwavefunction_' + param_string + ending + '.ckpt'
        with tf.variable_scope(wf.scope,reuse=tf.compat.v1.AUTO_REUSE):
            with wf.graph.as_default():
                saver.restore(sess,path+'/'+filename + steps_string)
                meanEnergy=np.load(save_dir + '/meanEnergy_' + param_string + savename + ending + '.npy').tolist()
                varEnergy=np.load(save_dir + '/varEnergy_'+ param_string + savename + ending + '.npy').tolist()

    with tf.compat.v1.variable_scope(wf.scope,reuse=tf.compat.v1.AUTO_REUSE):
        with wf.graph.as_default():

            Ns = 2 * Nx -2 #new samples per row after interaction
            samples_ = wf.sample(numsamples=numsamples,inputdim=input_dim)
            samples = np.ones((numsamples, Ny,Nx), dtype=np.int32)

            samples_placeholder=tf.compat.v1.placeholder(dtype=tf.int32,shape=(None,Ny,Nx))
            log_ampl_tensor=wf.log_amplitude(samples_placeholder,inputdim=input_dim)

            queue_samples = np.zeros((Ns*Ny+1, numsamples, Ny,Nx), dtype = np.int32) #Array to store all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)
            log_amplitudes = np.zeros((Ns*Ny+1)*numsamples, dtype=np.complex64) #Array to store the log_probs of all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)

            first_entropy_annealing = True
            if entropy:
                lambda0, entropy_steps = entropy_params
                delta_lambda = lambda0 * entropy_steps/(numsteps*0.5-entropy_steps)
                lambda_ = lambda0
            else:
                lambda_= 0.0
                entropy_steps = numsteps + 100
            
            
            
            starting_steps = len(meanEnergy)
            for it in range(len(meanEnergy),numsteps+1):

                samples=sess.run(samples_)

                #Update Entropy Annealing
                if (it - starting_steps)%entropy_steps == 1:
                    if first_entropy_annealing:
                        first_entropy_annealing = False
                    else:
                        lambda_ -= delta_lambda
                        if lambda_<1e-10:
                            lambda_ = 0.0
                        print("Lambda = ", lambda_)
        

                #Estimating local_energies
                local_energies = Kitaev2D_local_energies(J1_, J2_, J3_, Nx, Ny, samples, queue_samples, log_ampl_tensor, samples_placeholder, log_amplitudes, lambda_, batch_size, sess)

                meanE = np.mean(local_energies)
                varE = np.var(local_energies)

                #adding elements to be saved
                meanEnergy.append(meanE)
                varEnergy.append(varE)

                #lr_adaptation
                lr_adapted = lr*(1+it/5000)**(-1)
                #Optimize
                sess.run(optstep,feed_dict={Eloc:local_energies,samp:samples,learningrate_placeholder: lr_adapted})

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
