import scipy.io as io
from scipy.io import *
def saveDAGM():
    path = ['checkpoints/DAGM/WideResNet28_10_S2M2_R/last/Fsem_1.mat',
            ]
    for x in path:
        data = io.loadmat(x)
        features = data['features']
        labels = data['labels']
        unfeatures2 = features[700:, :]
        unlabels2 = labels[:, 700:]
        unfeatures3 = features[:700, :]
        unlabels3 = labels[:, :700]
        # unseen
        savemat("checkpoints/DAGM/WideResNet28_10_S2M2_R/last/Fsem_2.mat", {'features': unfeatures2, 'labels': unlabels2})
        # seen
        savemat("checkpoints/DAGM/WideResNet28_10_S2M2_R/last/Fsem.mat", {'features': unfeatures3, 'labels': unlabels3})
# saveDAGM()
def saveKTH():
    path = ['checkpoints/KTH/WideResNet28_10_S2M2_R/last/Fsem.mat',
            ]
    for x in path:
        data = io.loadmat(x)
        features = data['features']
        labels = data['labels']
        unfeatures2 = features[700:, :]
        unlabels2 = labels[:, 700:]
        unfeatures3 = features[:700, :]
        unlabels3 = labels[:, :700]
        # unseen
        savemat("checkpoints/KTH/WideResNet28_10_S2M2_R/last/Fsem_2.mat", {'features': unfeatures2, 'labels': unlabels2})
        # seen
        savemat("checkpoints/KTH/WideResNet28_10_S2M2_R/last/Fsem.mat", {'features': unfeatures3, 'labels': unlabels3})
#saveKTH()
def saveKTD():
    path = ['checkpoints/KTD/WideResNet28_10_S2M2_R/last/Fsem.mat',
            ]
    for x in path:
        data = io.loadmat(x)
        features = data['features']
        labels = data['labels']
        unfeatures2 = features[1500:, :]
        unlabels2 = labels[:, 1500:]
        unfeatures3 = features[:1500, :]
        unlabels3 = labels[:, :1500]
        # unseen
        savemat("checkpoints/KTD/WideResNet28_10_S2M2_R/last/Fsem_2.mat", {'features': unfeatures2, 'labels': unlabels2})
        # seen
        savemat("checkpoints/KTD/WideResNet28_10_S2M2_R/last/Fsem.mat", {'features': unfeatures3, 'labels': unlabels3})
# saveKTD()
def saveMSD():
    path = ['checkpoints/Aluminum/WideResNet28_10_S2M2_R/last/Fsem.mat',
            ]
    for x in path:
        data = io.loadmat(x)
        features = data['features']
        labels = data['labels']
        unfeatures2 = features[1500:, :]
        unlabels2 = labels[:, 1500:]
        unfeatures3 = features[:1500, :]
        unlabels3 = labels[:, :1500]
        # unseen
        savemat("checkpoints/Aluminum/WideResNet28_10_S2M2_R/last/Fsem_2.mat", {'features': unfeatures2, 'labels': unlabels2})
        # seen
        savemat("checkpoints/Aluminum/WideResNet28_10_S2M2_R/last/Fsem_3.mat", {'features': unfeatures3, 'labels': unlabels3})
saveMSD()