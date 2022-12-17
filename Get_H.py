#DAGM2007
# acc_unseen = [51.73, 52.67, 51.75, 51.58, 46.94,]
# acc_seen = [48.61,  56.24, 56.75, 57.32, 55.85,]
#KTH-TIPS
# acc_unseen = [20.03, 21.36, 21.76, 23.40, 25.09,]
# acc_seen = [75.04,  80.19, 81.74, 80.75, 77.27,]
#KTD
# acc_unseen = [27.65, 37.19, 42.95, 43.78, 41.55,]
# acc_seen = [56.07,  55.90, 55.56, 54.66, 50.66,]
#MSD
acc_unseen = [41.67, 17.22, 22.01, 12.28,]
acc_seen = [64.04,  75.71, 52.02, 35.88,]
for i in range(0, len(acc_seen)):
    H = 2*acc_seen[i]*acc_unseen[i] / (acc_seen[i]+acc_unseen[i])
    print(H)