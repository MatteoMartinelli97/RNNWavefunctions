import tensorflow as tf
import numpy as np
import random
import sys

def sqsoftmax(inputs):
   return tf.sqrt(tf.nn.softmax(inputs))

def softsign_(inputs):
    return np.pi*(tf.nn.softsign(inputs))

def heavyside(inputs):
    sign = tf.sign(tf.sign(inputs) + 0.1 ) #tf.sign(0) = 0, this is a way to cure that since I want tf.sign(0) = 1
    return 0.5*(sign+1.0)

class RNNwavefunction(object):
    def __init__(self,systemsize,inputdim = 4,cell=None,units=[10,10],scope='RNNwavefunction', seed=111):
        """
            systemsize:  int, size of the lattice
            cell:        a tensorflow RNN cell
            units:       list of int
                         number of units per RNN layer
            scope:       str
                         the name of the name-space scope
            seed:       pseudo-random number generator
        """
        self.graph=tf.Graph()
        self.scope=scope #Label of the RNN wavefunction
        self.N=systemsize #Number of sites of the 1D chain
        self.inputdim = inputdim #Hilbert space dimension
        
        #Seeding
        random.seed(seed)  # `python` built-in pseudo-random generator
        np.random.seed(seed)  # numpy pseudo-random generator

        #Defining the neural network
        with self.graph.as_default():
            with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):
              tf.set_random_seed(seed)  # tensorflow pseudo-random generator
              #Define the RNN cell where units[n] corresponds to the number of memory units in each layer n
              self.rnn=tf.nn.rnn_cell.MultiRNNCell([cell(units[n]) for n in range(len(units))])

              self.dense_ampl = tf.layers.Dense(4,activation=sqsoftmax,name='wf_dense_ampl') #Define the Fully-Connected layer followed by a square root of Softmax
              self.dense_phase = tf.layers.Dense(4,activation=softsign_,name='wf_dense_phase') #Define the Fully-Connected layer followed by a Softsign*pi

    def sample(self,numsamples, inputdim):
        """
            Generate samples from a probability distribution parametrized by a recurrent network
            ------------------------------------------------------------------------
            Parameters:
            numsamples:      int
                             number of samples to be produced
            inputdim:        int
                             hilbert space dimension of one spin
            ------------------------------------------------------------------------
            Returns:      
            samples:         tf.Tensor of shape (numsamples,systemsize)
                             the samples in integer encoding
        """
        with self.graph.as_default(): #Call the default graph, used if not willing to create multiple graphs.
            with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):

                a=np.zeros((numsamples,inputdim)).astype(np.float32)

                #Initial input sigma_0 
                #0 state in one-hot encoding for Hilbert space of dimension inputdim
                inputs=tf.constant(dtype=tf.float32,value=a,shape=[numsamples,inputdim])
                
                self.outputdim=self.inputdim
                self.numsamples=inputs.shape[0]

                samples=[]
                
                #Define the initial hidden state of the RNN
                rnn_state = self.rnn.zero_state(self.numsamples,dtype=tf.float32) 

                inputs_ampl = inputs

                for n in range(self.N):
                  rnn_output,rnn_state = self.rnn(inputs_ampl, rnn_state)

                  #Applying softmax layer for amplitudes
                  output_ampl = self.dense_ampl(rnn_output)
                
                  sample_temp=tf.reshape(tf.random.categorical(tf.log(output_ampl**2),num_samples=1),[-1,])
                  samples.append(sample_temp)
                  inputs=tf.one_hot(sample_temp,depth=self.outputdim)

                  inputs_ampl = inputs
                  tf.print(sample_temp, output_stream=sys.stdout)
        self.samples=tf.stack(values=samples,axis=1) # (self.N, num_samples) to (num_samples, self.N): Generate self.numsamples vectors of size self.N spin containing 0, 1, 2, or 3

        return self.samples

    """
    def log_amplitude(self,samples,inputdim):
    """
    """
            calculate the log-amplitudes of ```samples`` while imposing zero magnetization
            ------------------------------------------------------------------------
            Parameters:
            samples:         tf.Tensor
                             a tf.placeholder of shape (number of samples,systemsize)
                             containing the input samples in integer encoding
            inputdim:        int
                             dimension of the input space
            ------------------------------------------------------------------------
            Returns:
            log-amps      tf.Tensor of shape (number of samples,)
                             the log-amplitude of each sample
       
    """
    """
       with self.graph.as_default():

            self.inputdim=inputdim
            self.outputdim=self.inputdim

            self.numsamples=tf.shape(samples)[0]
            a=tf.zeros(self.numsamples, dtype=tf.float32)
            b=tf.zeros(self.numsamples, dtype=tf.float32)
            c=tf.zeros(self.numsamples, dtype=tf.float32)
            d=tf.zeros(self.numsamples, dtype=tf.float32)

            #Initial input sigma_0 
            #0 state in one-hot encoding for Hilbert space of dimension inputdim
            inputs=tf.stack([a,b,c,d], axis = 1)

            with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):
                amplitudes=[]

                rnn_state = self.rnn.zero_state(self.numsamples,dtype=tf.float32)

                inputs_ampl = inputs

                for n in range(self.N):

                    rnn_output,rnn_state = self.rnn(inputs_ampl, rnn_state)

                    #Applying softmax layer for amplitude
                    output_ampl = self.dense_ampl(rnn_output)
                    #Applying softsign layer for phase
                    output_phase = self.dense_phase(rnn_output)

                    #You can add a bias (I don't)
                    #compute the complex amplitude of the RNN
                    amplitude = tf.complex(output_ampl,0.0)*tf.exp(tf.complex(0.0,output_phase)) 
                    amplitudes.append(amplitude)

                    inputs=tf.reshape(tf.one_hot(tf.reshape(tf.slice(samples,begin=[np.int32(0),np.int32(n)],size=[np.int32(-1),np.int32(1)]),shape=[self.numsamples]),depth=self.outputdim),shape=[self.numsamples,self.inputdim])
                    inputs_ampl = inputs

            amplitudes=tf.stack(values=amplitudes,axis=1) # (self.N, num_samples,inputdim) to (num_samples, self.N, inputdim): Generate self.numsamples vectors of size (self.N, inputdim) spin containing the log_amplitudes of each sample
            one_hot_samples=tf.one_hot(samples,depth=self.inputdim)

            print("Amp = ", amplitudes)
            imp_val = tf.reduce_sum(tf.multiply(amplitudes,tf.complex(one_hot_samples,tf.zeros_like(one_hot_samples))),axis=2)
            self.log_amplitudes = tf.reduce_sum(tf.log(1E-13 + tf.reduce_sum(tf.multiply(amplitudes,tf.complex(one_hot_samples,tf.zeros_like(one_hot_samples))),axis=2)),axis=1) #To get the log amplitude of each sample

            return self.log_amplitudes, imp_val


    """

    def log_amplitude(self,samples,inputdim):
        """
            calculate the log-amplitudes of ```samples`` while imposing zero magnetization
            ------------------------------------------------------------------------
            Parameters:
            samples:         tf.Tensor
                             a tf.placeholder of shape (number of samples,systemsize)
                             containing the input samples in integer encoding
            inputdim:        int
                             dimension of the input space
            ------------------------------------------------------------------------
            Returns:
            log-amps      tf.Tensor of shape (number of samples,)
                             the log-amplitude of each sample
            """
        with self.graph.as_default():

            self.inputdim=inputdim
            self.outputdim=self.inputdim

            self.numsamples=tf.shape(samples)[0]
            a=tf.zeros(self.numsamples, dtype=tf.float32)
            b=tf.zeros(self.numsamples, dtype=tf.float32)
            c=tf.zeros(self.numsamples, dtype=tf.float32)
            d=tf.zeros(self.numsamples, dtype=tf.float32)

            #Initial input sigma_0 
            #0 state in one-hot encoding for Hilbert space of dimension inputdim
            inputs=tf.stack([a,b,c,d], axis = 1)

            with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):
                amplitudes=[]
                phases = []
                rnn_state = self.rnn.zero_state(self.numsamples,dtype=tf.float32)

                inputs_ampl = inputs

                for n in range(self.N):

                    rnn_output,rnn_state = self.rnn(inputs_ampl, rnn_state)

                    #Applying softmax layer for amplitude
                    output_ampl = self.dense_ampl(rnn_output)
                    #Applying softsign layer for phase
                    output_phase = self.dense_phase(rnn_output)

                    #You can add a bias (I don't)
                    #compute the complex amplitude of the RNN
                    amplitude = output_ampl
                    phase = output_phase
                    amplitudes.append(amplitude)
                    phases.append(phase)
                    
                    

                    inputs=tf.reshape(tf.one_hot(tf.reshape(tf.slice(samples,begin=[np.int32(0),np.int32(n)],size=[np.int32(-1),np.int32(1)]),shape=[self.numsamples]),depth=self.outputdim),shape=[self.numsamples,self.inputdim])
                    inputs_ampl = inputs

            # (self.N, num_samples,inputdim) --> (num_samples, self.N, inputdim)
            amplitudes=tf.stack(values=amplitudes,axis=1) 
            phases=tf.stack(values=phases,axis=1)
    

            one_hot_samples=tf.one_hot(samples,depth=self.inputdim)

            log_amp = tf.reduce_sum(tf.math.log( tf.add(1E-13, tf.reduce_sum( tf.math.multiply(amplitudes, one_hot_samples),axis=2 )) ),axis=1) #To get the log amplitude of each sample
            log_pha = tf.reduce_sum( tf.reduce_sum( tf.math.multiply(phases, one_hot_samples),axis=2 ) ,axis=1)
        
            self.log_amplitudes = tf.complex(log_amp, log_pha) #tf.zeros_like(log_amp))

        return self.log_amplitudes