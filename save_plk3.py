import pickle
import scipy.io as io
Wrn = io.loadmat("./checkpoints/Aluminum/WideResNet28_10_S2M2_R/last/Wrn.mat")
Fsem = io.loadmat("./checkpoints/Aluminum/WideResNet28_10_S2M2_R/last/Fsem.mat")

Wrn['labels'] = Wrn['labels'].reshape(Wrn['labels'].shape[1])
Wrn_dict = {'data': Wrn['features'], 'labels': Wrn['labels']}

# pickle a variable to a file
file = open('Wrn.plk', 'wb')
pickle.dump(Wrn_dict, file)
file.close()