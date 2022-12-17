import scipy.io as io
path = ['checkpoints/KTH/WideResNet28_10_S2M2_R/last/Fsem.mat', 'checkpoints/KTH/WideResNet28_10_S2M2_R/last/Wrn.mat',
        'checkpoints/KTH/WideResNet28_10_S2M2_R/last/Fsem_2.mat', 'checkpoints/KTH/WideResNet28_10_S2M2_R/last/Wrn_2.mat']
for x in path:
    data = io.loadmat(x)
    print(x)
    print(data)
    print(len(data))
    print('===================================================================================================================')