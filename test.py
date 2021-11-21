import cv2
import numpy as np
import os
#import pandas as pd
import random as rd
import math
import sys
import matplotlib.pyplot as plt

## Training ##
path = 'images/train1/original/'
path2 = 'images/train1/gray/'
path3 = 'images/train1/normalized/'
files = []
meansl = []
for r, d, f in os.walk(path):
    for file in f:
        if('.png' in file or '.jpg' in file):
            files.append(os.path.join(r, file))
for input_file in files:
    f = cv2.imread(input_file)
    # print(f.shape)
    a = (f[:, :, 0]*0.2126+f[:, :, 1]*0.0722+f[:, :, 2]*0.7152)
    f[:, :, 0] = a  # normal : 0.114*B+0.587*G+0.2989*R
    f[:, :, 1] = a
    f[:, :, 2] = a
    # print(f)
    f = (255/1)*(f/(255/1))**2  # increase different between sky and cloud
    f = f.astype(np.uint8)
    # print(f)
    cv2.imwrite(path2+'gray_'+input_file[-8:],
                cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    mean_all_pixels = np.mean(f)  # find mean of all pixels
    _, n = cv2.threshold(f, mean_all_pixels, 255,
                         cv2.THRESH_TOZERO)  # thresholding
    check = (n[:, :, 0] != 0)
    sum1 = np.sum(n[check])/3  # get only one channel
    z = (check == 1).sum()
#    plt.imshow(n,cmap='gray')
#    plt.show()
    cv2.imwrite(path3+'normalized_' +
                input_file[-8:], cv2.cvtColor(n, cv2.COLOR_BGR2GRAY))
    mean_cloud = sum1/z  # find mean of cloud area
    print(input_file+" mean = "+str(mean_cloud))
    meansl.append(mean_cloud)
M = max(meansl)
m = min(meansl)


def InitializeMeans(k, m, M):
    means = [[0] for _ in range(k)]  # [[0],[0],...] with only 1 feature
    for item in means:
        item[0] = rd.uniform(m+1, M-1)  # random mean for initial mean
    return means


def EuclideanDistance(x, y):
    S = math.pow(x-y, 2)
    return math.sqrt(S)


def UpdateMean(n, mean, item):
    m2 = mean[0]
    m2 = (m2*(n-1)+item)/float(n)
    mean[0] = round(m2, 3)
    return mean


def CalculateMeans(k, maxIterations=100000):
    cMin = m
    cMax = M
    # Initialize means at random points
    means = InitializeMeans(k, cMin, cMax)
    # Initialize clusters, the array to hold
    # the number of items in a class
    clusterSizes = [0]*len(means)
    # An array to hold the cluster an item is in
    belongsTo = [0]*len(meansl)
    # Calculate means
    for _ in range(maxIterations):  # loop until no more cluster change
        # If no change of cluster occurs, halt
        print(means)
        noChange = True
        for i in range(len(meansl)):  # meansl: each cloud mean of training set
            item = meansl[i]
            # Classify item into a cluster and update the
            # corresponding means.
            index = Classify(means, item)  # check distance with each cluster
            print(index)
            clusterSizes[index] += 1
            print(clusterSizes)
            cSize = clusterSizes[index]
            print(means[index])
            means[index] = UpdateMean(
                cSize, means[index], item)  # updated cluster mean
            print(means)
            # Item changed cluster
            if (index != belongsTo[i]):
                noChange = False
            belongsTo[i] = index
            print(belongsTo)
            # Nothing changed, return
        if (noChange):
            break
    return means


def Classify(means, item):
    minimum = sys.maxsize
    index = -1
    for i in range(len(means)):  # means: mean of each cluster
        print("i:"+str(i))
        dis = EuclideanDistance(item, means[i][0])
        print("dis:"+str(dis))
        if (dis < minimum):
            minimum = dis
            index = i
    return index


def FindClusters(means, meansl):
    clusters = [[] for _ in range(len(means))]  # Initialize clusters
    for item in meansl:  # meansl: each cloud mean of training set
        index = Classify(means, item)
        clusters[index].append(item)
    return clusters


means = CalculateMeans(3)  # Calculate 3 cluster means
means.sort()
print("Means = "+str(means))
g = FindClusters(means, meansl)
print("Cluster = "+str(g))

############################### Testing #################################
# path = 'images/test/original/'
# path2 = 'images/test/gray/'
# path3 = 'images/test/normalized/'
# files = []
# for r, d, f in os.walk(path):
#     for file in f:
#         if('.png' in file or '.jpg' in file):
#             files.append(os.path.join(r, file))
# for input_file in files:
#     m = cv2.imread(input_file)
#     v = 0
#     # normal : 0.114*B+0.587*G+0.2989*R (increase B and R ratio)
#     m[:, :, 0] = m[:, :, 0]*0.2126+m[:, :, 1]*0.0722+m[:, :, 2]*0.7152
#     m[:, :, 1] = m[:, :, 0]*0.2126+m[:, :, 1]*0.0722+m[:, :, 2]*0.7152
#     m[:, :, 2] = m[:, :, 0]*0.2126+m[:, :, 1]*0.0722+m[:, :, 2]*0.7152
#     m = (255/1) * (m/(255/1))**2
#     m = m.astype(np.uint8)
# #    plt.imshow(m,cmap='gray')
# #    plt.show()
#     a = input_file.split('/')
#     cv2.imwrite(path2+'gray_'+a[3], cv2.cvtColor(m, cv2.COLOR_BGR2GRAY))
#     mean_all_pixels = np.mean(m)  # find mean of all pixels
#     _, r = cv2.threshold(m, mean_all_pixels, 255,
#                          cv2.THRESH_TOZERO)  # thresholding
#     check = (r[:, :, 0] != 0)
#     sum1 = np.sum(r[check])/3
#     z = (check == 1).sum()
# #    plt.imshow(r,cmap='gray')
# #    plt.show()
#     cv2.imwrite(path3+'normalized_'+a[3], cv2.cvtColor(r, cv2.COLOR_BGR2GRAY))
#     mean_cloud = sum1/z  # find mean of cloud area
#     print(input_file+" mean = "+str(mean_cloud))
#     for i in range(len(means)):
#         if mean_cloud > means[i]:
#             v = i
#     if v != len(means)-1:
#         o1 = means[v+1]-mean_cloud
#         o2 = mean_cloud-means[v]
#         if o1 < o2:
#             v = v+1
#     else:
#         v = len(means)-1
#     print('Current weather condition is:')
#     if v == 0:
#         print('SUNNY')
#     elif v == 1:
#         print('CLOUDY')
#     elif v == 2:
#         print('HIGH CHANCE OF RAIN')
