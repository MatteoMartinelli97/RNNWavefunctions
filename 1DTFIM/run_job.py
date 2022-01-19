from TrainingRNN_1DTFIM import run_1DTFIM
import os


#system size
N = 80
#set of num_units
units = [1]
seed = 111
#set different fields
Bs = [1]
numsamples = 200

os.chdir('Critical')
if not os.path.exists('seed{}'.format(seed)):
      os.mkdir('seed{}'.format(seed))
os.chdir('seed{}'.format(seed))


for u in units:
  for B in Bs:
    path = '{0}u_22ksteps_seed{1}'.format(u, seed)
    if not os.path.exists(path):
      os.mkdir(path)

    x, y = run_1DTFIM(numsteps = 22000, systemsize = N, Bx = +B, 
                                     num_units = u,  num_layers = 1, numsamples = numsamples, 
                                     learningrate = 5e-3, seed = seed,
                                     symRNN = False, load_model = False, 
                                     #save_dir = 'gdrive/My Drive/RNNWavefunctions-master/Results/1DTFIM/Under_Critical/{0}u_{1:.2f}B'.format(u, B),
                                    save_dir = 'gdrive/My Drive/RNNWavefunctions-master/Results/1DTFIM/Trained_Models/seed_{1}/{0}u_22ksteps_seed{1}'.format(u, seed),
                                    #save_dir = 'gdrive/My Drive/RNNWavefunctions-master/Results/1DTFIM/Trained_Models/6u/6u_seed{}'.format(seed),
                                    #save_dir = 'gdrive/My Drive/RNNWavefunctions-master/Results/1DTFIM/Above_Critical/{0}u_{1:.2f}B'.format(u, B),
                                    #save_dir = 'gdrive/My Drive/RNNWavefunctions-master/Results/1DTFIM/Trained_Models/SpinFlipSymm/{0}u_22ksteps_seed{1}'.format(u, seed),
                                     checkpoint_steps = 5000
                                    )
