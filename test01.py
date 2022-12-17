# import FSLTask1
# #将特征输入ZSLGAN查看结果(特征矩阵转换)
# FSLTask1.loadDataSet("Aluminum")


# from scipy.spatial.distance import cosine
#
# vec1 = [1, 2, 3]
# vec2 = [2, 3, 4]
# vec3 = [1, 2, 3]
# s1 = 1-cosine(vec1, vec2)
# s2 = 1-cosine(vec1, vec3)
#
# print(s1)
# print(s2)
#索引排序
# num_list = [1, 8, 2, 3, 10, 4, 5]
# ordered_list = sorted(range(len(num_list)), key=lambda k: num_list[k])
# print(ordered_list)    # [0, 2, 3, 5, 6, 1, 4]
import torch

import filter_sem
shot = 0
data = "KTH"
# 广义混合未见类与可见类
if(data == "DAGM" or data == "KTH"):
        way = 10
if (data == "KTD" or data == "MSD"):
        way = 20
# way = 10 #广义未见类划分下，分类分别为10和20，传统未见类划分下，语义way = 实际分类n_way
dealsem = filter_sem.dealSem(shot, data, way)

print(dealsem)