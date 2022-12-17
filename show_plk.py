import numpy as np
import scipy.io as sio
# from termcolor import cprint
import pickle
import sys


_datasetFeaturesFiles = {"MSD_train1": "./checkpoints/Aluminum/WideResNet28_10_S2M2_R/last/output1.plk",
                         "MSD_train2": "./checkpoints/Aluminum/WideResNet28_10_S2M2_R/last/output2.plk",
                         "MSD": "./Wrn.plk",
                         "DAGM_10": "./checkpoints/DAGM/WideResNet28_10_S2M2_R/last/output_10.plk",
                         "DAGM_3": "./checkpoints/DAGM/WideResNet28_10_S2M2_R/last/output_3.plk",
                         "KTH_10": "./checkpoints/KTH/WideResNet28_10_S2M2_R/last/output_10.plk",
                         "KTH_3": "./checkpoints/KTH/WideResNet28_10_S2M2_R/last/output_3.plk",
                         "KTD_20": "./checkpoints/KTD/WideResNet28_10_S2M2_R/last/output_20.plk",
                         "KTD_5": "./checkpoints/KTD/WideResNet28_10_S2M2_R/last/output_5.plk",
                         }

def _load_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
        labels = [np.full(shape=len(data[key]), fill_value=key)
                  for key in data]
        data = [features for key in data for features in data[key]]
        getFeat = dict()
        getFeat['data'] = np.stack(data, axis=0)
        getFeat['labels'] = np.concatenate(labels)
        return getFeat
# 获取图片训练特征
getFeat1 = _load_pickle(_datasetFeaturesFiles['MSD_train2'])
blobs1 = getFeat1
pfc_feat_data_train = blobs1['data']  # image data
labels_train = blobs1['labels'].astype(int)  # class labels
print("=======================生成文件====================================")
print('getFeat1', getFeat1)
print("===============================训练特征====================================")
print('pfc_feat_data_train',pfc_feat_data_train)
print(pfc_feat_data_train.shape)
print('labels_train',labels_train)
print(labels_train.shape)