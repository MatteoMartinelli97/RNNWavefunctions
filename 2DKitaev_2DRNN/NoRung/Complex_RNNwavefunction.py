import tensorflow as tf
import numpy as np
import random

def sqsoftmax(inputs):
   return tf.sqrt(tf.nn.softmax(inputs))

def softsign_(inputs):
    return np.pi*(tf.nn.softsign(inputs))

class RNNwavefunction(object):
    def __init__(self,systemsize_x, systemsize_y, hilbert_dim = 2, cell=None,units=[10],scope='RNNwavefunction',seed = 111):
        """
            systemsize_x:  int
                         number of sites for x-axis
            systemsize_y:  int
                         number of sites for y-axis         
            cell:        a tensorflow RNN cell
            units:       list of int
                         number of units per RNN layer
            scope:       str
                         the name of the name-space scope
            seed:        pseudo-random number generator 
        """
        self.graph=tf.Graph()
        self.scope=scope #Label of the RNN wavefunction
        self.Nx=systemsize_x #size of x direction in the 2d model
        self.Ny=systemsize_y

        random.seed(seed)  # `python` built-in pseudo-random generator
        np.random.seed(seed)  # numpy pseudo-random generator

        #Defining the neural network
        with self.graph.as_default():
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):

              tf.compat.v1.set_random_seed(seed)  # tensorflow pseudo-random generator
              self.rnn=cell(num_units = units[0], num_in = hilbert_dim ,name="rnn_"+str(0),dtype=tf.float32)
              self.dense_amp = tf.compat.v1.layers.Dense(hilbert_dim,activation=sqsoftmax,name='wf_dense', dtype = tf.float32)
              self.dense_pha = tf.compat.v1.layers.Dense(hilbert_dim,activation=tf.nn.relu,name='wf_phase', dtype = tf.float32)

    def sample(self,numsamples,inputdim):
        """
            generate samples from a probability distribution parametrized by a recurrent network
            ------------------------------------------------------------------------
            Parameters:

            numsamples:      int
                             
                             
                             
                             samples to be produced
            inputdim:        int
                             hilbert space dimension of one spin

            ------------------------------------------------------------------------
            Returns:      

            samples:         tf.Tensor of shape (numsamples,systemsize_y, systemsize_x)
                             the samples in integer encoding
        """

        with self.graph.as_default(): #Call the default graph, used if not willing to create multiple graphs.
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):

                #Initial input to feed to the 2drnn

                self.inputdim=inputdim
                self.outputdim=self.inputdim
                self.numsamples=numsamples


                samples=[[[] for nx in range(self.Nx)] for ny in range(self.Ny)]
                rnn_states = {}
                inputs = {}

                for ny in range(self.Ny): #Loop over the rows
                    if ny%2==0: #set left boundary for even rows
                        nx = -1
                        rnn_states[str(ny)+str(nx)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32)
                        inputs[str(ny)+str(nx)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32) 

                    if ny%2==1:
                        nx = self.Nx #set right boundary for odd rows
                        rnn_states[str(ny)+str(nx)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32)
                        inputs[str(ny)+str(nx)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32) 


                for nx in range(self.Nx): #Loop over the top row and add ppadding
                    ny = -1
                    rnn_states[str(ny)+str(nx)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32)
                    inputs[str(ny)+str(nx)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32) 

                #Begin sampling
                for ny in range(self.Ny): 

                    if ny%2 == 0:

                        for nx in range(self.Nx): #left to right

                            rnn_output, rnn_states[str(ny)+str(nx)] = self.rnn((inputs[str(ny)+str(nx-1)],inputs[str(ny-1)+str(nx)]), (rnn_states[str(ny)+str(nx-1)],rnn_states[str(ny-1)+str(nx)]))

                            ampl=self.dense_amp(rnn_output)
                            sample_temp=tf.reshape(tf.random.categorical(tf.compat.v1.log(ampl**2),num_samples=1),[-1,])
                            samples[ny][nx] = sample_temp
                            inputs[str(ny)+str(nx)]=tf.one_hot(sample_temp,depth=self.outputdim, dtype = tf.float32)


                    if ny%2 == 1:

                        for nx in range(self.Nx-1,-1,-1): #right to left

                            rnn_output, rnn_states[str(ny)+str(nx)] = self.rnn((inputs[str(ny)+str(nx+1)],inputs[str(ny-1)+str(nx)]), (rnn_states[str(ny)+str(nx+1)],rnn_states[str(ny-1)+str(nx)]))

                            ampl=self.dense_amp(rnn_output)
                            sample_temp=tf.reshape(tf.random.categorical(tf.compat.v1.log(ampl**2),num_samples=1),[-1,])
                            samples[ny][nx] = sample_temp
                            inputs[str(ny)+str(nx)]=tf.one_hot(sample_temp,depth=self.outputdim, dtype = tf.float32)


            self.samples=tf.transpose(tf.stack(values=samples,axis=0), perm = [2,0,1]) #give the wanted output shape

        return self.samples

    def log_amplitude(self,samples,inputdim):
        """
            calculate the log-amplitudes of ```samples``
            ------------------------------------------------------------------------
            Parameters:

            samples:         tf.Tensor
                             a tf.placeholder of shape (number of samples,systemsize_y,system_size_x)
                             containing the input samples in integer encoding
            inputdim:        int
                             dimension of the input space

            ------------------------------------------------------------------------
            Returns:
            log-ampl        tf.Tensor of shape (number of samples,)
                             the log-amplitude of each sample
            """
        with self.graph.as_default():

            self.inputdim=inputdim
            self.outputdim=self.inputdim
            self.numsamples=tf.shape(samples)[0]
            self.outputdim=self.inputdim


            samples_=tf.transpose(samples, perm = [1,2,0]) #(Ny, Nx, numsamples)
            rnn_states = {}
            inputs = {}

            for ny in range(self.Ny): #Loop over the boundary
                if ny%2==0:
                    nx = -1
                    rnn_states[str(ny)+str(nx)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32)
                    inputs[str(ny)+str(nx)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32) 

                if ny%2==1:
                    nx = self.Nx
                    rnn_states[str(ny)+str(nx)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32)
                    inputs[str(ny)+str(nx)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32) 


            for nx in range(self.Nx): #Loop over the boundary
                ny = -1
                rnn_states[str(ny)+str(nx)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32)
                inputs[str(ny)+str(nx)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32) 


            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
                amplitudes = [[[] for nx in range(self.Nx)] for ny in range(self.Ny)]
                phases = [[[] for nx in range(self.Nx)] for ny in range(self.Ny)]

                #Begin estimation of log probs
                for ny in range(self.Ny):

                    if ny%2 == 0:

                        for nx in range(self.Nx): #left to right

                            rnn_output, rnn_states[str(ny)+str(nx)] = self.rnn((inputs[str(ny)+str(nx-1)],inputs[str(ny-1)+str(nx)]), (rnn_states[str(ny)+str(nx-1)],rnn_states[str(ny-1)+str(nx)]))

                            ampl=self.dense_amp(rnn_output)
                            pha=self.dense_pha(rnn_output)
                            amplitudes[ny][nx] = ampl
                            phases[ny][nx]=pha
                            inputs[str(ny)+str(nx)]=tf.one_hot(samples_[ny,nx],depth=self.outputdim,dtype = tf.float32)

                    if ny%2 == 1:

                        for nx in range(self.Nx-1,-1,-1): #right to left

                            rnn_output, rnn_states[str(ny)+str(nx)] = self.rnn((inputs[str(ny)+str(nx+1)],inputs[str(ny-1)+str(nx)]), (rnn_states[str(ny)+str(nx+1)],rnn_states[str(ny-1)+str(nx)]))

                            ampl=self.dense_amp(rnn_output)
                            pha=self.dense_pha(rnn_output)
                            amplitudes[ny][nx] = ampl
                            phases[ny][nx]=pha
                            inputs[str(ny)+str(nx)]=tf.one_hot(samples_[ny,nx],depth=self.outputdim,dtype = tf.float32)
            
            #back to (numsamples, Ny, Nx, hilbert_dim)
            amplitudes=tf.transpose(tf.stack(values=amplitudes,axis=0),perm=[2,0,1,3]) 
            phases=tf.transpose(tf.stack(values=phases,axis=0),perm=[2,0,1,3])
            one_hot_samples=tf.one_hot(samples,depth=self.inputdim, dtype = tf.float32) 

            #Sum is over 3 axis: 3 is the one-hot vector, 2 and 1 are x and y (sum over all the spins) ->
            #the only axis left is 0 which are the different samples
            log_amp=tf.reduce_sum(tf.reduce_sum(tf.compat.v1.log(tf.reduce_sum(tf.multiply(amplitudes,one_hot_samples),axis=3)),axis=2),axis=1)
            log_pha=tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.multiply(phases,one_hot_samples),axis=3),axis=2),axis=1)

            self.log_amplitudes = tf.complex(log_amp, log_pha)
            
            return self.log_amplitudes
