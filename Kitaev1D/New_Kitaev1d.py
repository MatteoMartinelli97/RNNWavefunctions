##################
## NEW WAY
##################

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #stop displaying tensorflow warnings
import numpy as np
import os
import time
import random
from math import ceil

from ComplexRNNwavefunction import RNNwavefunction

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
                matrixelements[num] =  sign * j2[site] * 0.25
                num += 1
        
        else:
            if j2[site] != 0.0:
                sig=np.copy(sigmap)
                sig[site]  = (2 + sigmap[site]) % 4
                sig[site +1]  = (2 + sigmap[site +1]) % 4
                sigmaH[num] = sig
                sign = 1 - 2 * ( (sig[site] == sig[site+1]) |  ((sig[site] + sig[site+1])%4 == 1) )
                matrixelements[num] =  sign * j2[site] * 0.25
                num += 1
            
            if j1[site] != 0.0:
                sig=np.copy(sigmap)
                sig[site] += 1 - 2 * (sig[site] % 2)
                sig[site+1] += 1 - 2 * (sig[site+1] % 2)
                sigmaH[num] = sig
                matrixelements[num] = j1[site] * 0.25
                num += 1

    return num

def KitaevSlices(j1, j2, j3, sigmasp, sigmas, H, sigmaH, matrixelements):
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
        num = KitaevMatrixElements(j1, j2, j3, sigmap, sigmaH, matrixelements)#note that sigmas[0,:]==sigmap, matrixelements and sigmaH are updated
        slices.append(slice(sigmas_length,sigmas_length + num))
        s = slices[n]

        H[s] = matrixelements[:num]
        sigmas[s] = sigmaH[:num]

        sigmas_length += num #Increasing the length of matrix elements sigmas

    return slices, sigmas_length


def run_Kitaev(numsteps = 10**5, systemsize = 20, j1_  = 1.0, j2_ = 1.0, j3_ = 1.0,
            num_units = 50, num_layers = 1, numsamples = 500, 
            learningrate = 2.5*1e-4, seed = 111,
            save_dir = ".",  checkpoint_steps = 11000):

    N=systemsize #Number of spins
    lr = np.float64(learningrate)
    
    j1 = +j1_*np.ones(N)
    j2 = +j2_*np.ones(N)
    j3 = +j3_*np.ones(N)
    
    #Seeding
    tf.reset_default_graph()
    #random.seed(seed)  # `python` built-in pseudo-random generator
    #np.random.seed(seed)  # numpy pseudo-random generator
    #tf.set_random_seed(seed)  # tensorflow pseudo-random generator


    # Intitializing the RNN-----------
    units=[num_units]*num_layers #list containing the number of hidden units for each layer of the networks

    input_dim=4 #Dimension of the Hilbert space for each site (here = 2, up or down)
    numsamples_=20 #only for initialization; later I'll use a much larger value (see below)
    wf=RNNwavefunction(N, inputdim=4, units=units,cell=tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell, seed = seed) #contains the graph with the RNNs
    #contains the graph with the RNNs
    sampling=wf.sample(numsamples_,input_dim) #call this function once to create the dense layers


    with wf.graph.as_default(): #now initialize everything
        inputs=tf.placeholder(dtype=tf.int32,shape=[numsamples_,N]) #the inputs are the samples of all of the spins
        #defining adaptive learning rate
        global_step = tf.Variable(0, trainable=False)
        learningrate_placeholder=tf.placeholder(dtype=tf.float32,shape=[])
        learningrate_withexpdecay = tf.train.exponential_decay(learningrate_placeholder, global_step, decay_steps = 100, decay_rate = 1.0, staircase=True) #Adaptive Learning (decay_rate = 1 -> no decay)
        amplitudes=wf.log_amplitude(inputs,input_dim) #The probs are obtained by feeding the sample of spins.
        optimizer=tf.train.AdamOptimizer(learning_rate=learningrate_withexpdecay, beta1=0.9, beta2 = 0.999, epsilon = 1e-8)
        init=tf.global_variables_initializer()
    # End Intitializing

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
    #     print(variables_names)
        sum = 0
        values = sess.run(variables_names)
        for k,v in zip(variables_names, values):
            v1 = tf.reshape(v,[-1])
            print(k,v1.shape)
            sum +=v1.shape[0]
        print('The number of params is {0}'.format(sum))

    #Running the training -------------------

    path=os.getcwd()

    ending='_units'
    for u in units:
        ending+='_{0}'.format(u)
    
    param_string = 'N{0}_samp{4}_J1{1:.1f}_J2{2:.1f}_J3{3:.1f}_GRURNN_OBC'.format(N, j1_, j2_, j3_, numsamples)
    
    filename= save_dir + '/RNNwavefunction_' + param_string + ending + '.ckpt'
    savename = '_KITA'

    with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
        with wf.graph.as_default():
            Eloc=tf.placeholder(dtype=tf.complex64,shape=[numsamples])
            samp=tf.placeholder(dtype=tf.int32,shape=[numsamples,N])
            log_amplitudes_=wf.log_amplitude(samp,inputdim=input_dim)

            #now calculate the fake cost function: https://stackoverflow.com/questions/33727935/how-to-use-stop-gradient-in-tensorflow
            cost = 2*tf.real(tf.reduce_mean(tf.conj(log_amplitudes_)*tf.stop_gradient(Eloc)) - tf.conj(tf.reduce_mean(log_amplitudes_))*tf.reduce_mean(tf.stop_gradient(Eloc)))
            #Calculate Gradients---------------

            gradients, variables = zip(*optimizer.compute_gradients(cost))

            #End calculate Gradients---------------

            optstep=optimizer.apply_gradients(zip(gradients,variables),global_step=global_step)
            sess.run(tf.variables_initializer(optimizer.variables()))

            saver=tf.train.Saver() #define tf saver


    meanEnergy=[]
    varEnergy=[]

    # #Loading previous trainings----------
    # with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
    #     with wf.graph.as_default():
    #         saver.restore(sess,path+''+filename)
    #         meanEnergy = np.load('../Check_Points/J1J2/meanEnergy_N'+str(N)+'_samp'+str(numsamples)+'_lradap'+str(lr)+'_complexGRURNN'+ savename + ending +'_zeromag.npy').tolist()
    #         varEnergy = np.load('../Check_Points/J1J2/varEnergy_N'+str(N)+'_samp'+str(numsamples)+'_lradap'+str(lr)+'_complexGRURNN'+ savename + ending +'_zeromag.npy').tolist()
    ## -----------
    #Running The training

    with tf.variable_scope(wf.scope,reuse=tf.AUTO_REUSE):
        with wf.graph.as_default():
            # max_grad = tf.reduce_max(tf.abs(gradients[0]))

            samples_ = wf.sample(numsamples=numsamples,inputdim=input_dim)
            samples = np.ones((numsamples, N), dtype=np.int32)

            inputs=tf.placeholder(dtype=tf.int32,shape=(None,N))
            log_amps=wf.log_amplitude(inputs,inputdim=input_dim)

            local_energies = np.zeros(numsamples, dtype = np.complex64) #The type complex should be specified, otherwise the imaginary part will be discarded

            sigmas=np.zeros((2*N*numsamples,N), dtype=np.int32) #Array to store all the diagonal and non diagonal sigmas for all the samples (We create it here for memory efficiency as we do not want to allocate it at each training step)
            H = np.zeros(2*N*numsamples, dtype=np.float32) #Array to store all the diagonal and non diagonal matrix elements for all the samples (We create it here for memory efficiency as we do not want to allocate it at each training step)
            log_amplitudes = np.zeros(2*N*numsamples, dtype=np.complex64) #Array to store all the diagonal and non diagonal log_probabilities for all the samples (We create it here for memory efficiency as we do not want to allocate it at each training step)

            sigmaH = np.zeros((2*N,N), dtype = np.int32) #Array to store all the diagonal and non diagonal sigmas for each sample sigma
            matrixelements=np.zeros(2*N, dtype = np.float32) #Array to store all the diagonal and non diagonal matrix elements for each sample sigma (the number of matrix elements is bounded by at most 2N)

            for it in range(len(meanEnergy),numsteps+1):

                #print("sampling started")

                #start = time.time()

                samples=sess.run(samples_)

                #end = time.time()
                #print("sampling ended: "+ str(end - start))

                #print("Magnetization = ", np.mean(2*samples - 1))

                #Getting the sigmas with the matrix elements
                slices, len_sigmas = KitaevSlices(j1, j2, j3,samples, sigmas, H, sigmaH, matrixelements)

                #Process in steps to get log amplitudes
                # print("Generating log amplitudes Started")
                #start = time.time()
                steps = ceil(len_sigmas/30000) #Process the sigmas in steps to avoid allocating too much memory

                # print("number of required steps :" + str(steps))

                for i in range(steps):
                    if i < steps-1:
                        cut = slice((i*len_sigmas)//steps,((i+1)*len_sigmas)//steps)
                    else:
                        cut = slice((i*len_sigmas)//steps,len_sigmas)

                    log_amplitudes[cut] = sess.run(log_amps,feed_dict={inputs:sigmas[cut]})
                    # print(i+1)
                    
                #end = time.time()
                # print("Generating log amplitudes ended "+ str(end - start))

                #Generating the local energies
                for n in range(len(slices)):
                    s=slices[n]
                    local_energies[n] = H[s].dot(np.exp(log_amplitudes[s]-log_amplitudes[s][0]))

                meanE = np.mean(local_energies)
                varE = np.var(np.real(local_energies))

                #adding elements to be saved
                meanEnergy.append(meanE)
                varEnergy.append(varE)
                #without learning decay
                sess.run(optstep,feed_dict={Eloc:local_energies,samp:samples,learningrate_placeholder: lr})

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