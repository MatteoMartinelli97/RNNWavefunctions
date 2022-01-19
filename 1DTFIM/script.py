import TrainingRNN_1DTFIM as tr
z1, x1, E1, var_E1 = tr.compute_correlations_from_model(numsamples = 25*10**4, old_numsamples=500, systemsize = 80, Bx = +1,
                                         num_units = 50,  num_layers = 1,
                                         batch_size = 25000, seed = 111,
                                         symRNN = False,
                                         save_dir = '../Results/Correlations/500samp'
                                        )
