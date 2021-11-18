import cv2
import numpy as np
import os
import pandas as pd
import random as rd,math
import sys
import matplotlib.pyplot as plt

#stdoutOrigin=sys.stdout 
#sys.stdout = open("log.txt", "w")

## Training ##
path='images/train/'
files=[]
meansl=[]
m=[]
M=[]
c=0
for r,d,f in os.walk(path):
    for file in f:
        if('.png' in file or '.jpg' in file):
            files.append(os.path.join(r,file))
            c+=1
#print(files)
for input_file in files:
    f=cv2.imread(input_file)
    h, w, bpp = np.shape(f)
    for py in range(0, h):
        for px in range(0, w):
            x = float(f[py][px][0])
            y = float(f[py][px][1])
            z = float(f[py][px][2])
            f[py][px] = x * 0.2126 + y * 0.0722 + z * 0.7152
#    plt.imshow(f,cmap='gray')
#    plt.show()
#    print(f)
    f = (255 / 1) * (f / (255 / 1)) ** 2
#    print(f)
    h, w, bpp = np.shape(f)
    n = np.zeros([h, w, bpp], dtype=np.uint8)
    c = 0
    sum = 0
    for py in range(0, h):
        for px in range(0, w):
            sum += f[py][px][0]
            c += 1
    for py in range(0, h):
        for px in range(0, w):
            if f[py][px][0] > (sum / c):
                n[py][px] = f[py][px]
                f[py][px] = 255
    sum1 = 0
    z = 0
    for py in range(0, h):
        for px in range(0, w):
            if n[py][px][0] != 0: # have a value (cloud area)
                sum1 += n[py][px][0]
                z += 1
#    plt.imshow(n,cmap='gray')
#    plt.show()
    mean = sum1 / z
    dict={}
    dict1={}
    dict[input_file]=mean
    dict1=mean
    print(input_file+" mean = "+str(dict1))
    meansl.append(dict1)
#print(meansl)
M.append(max(meansl))
m.append(min(meansl))

def InitializeMeans(meansl, k, m, M):
    f=1;#no. of features
    means=[[0 for i in range(f)] for j in range(k)];
    for item in means:
        for j in range(len(item)):
            item[j]= rd.uniform(m[j]+1,M[j]-1);
    return means;

def EuclideanDistance(x,y):
    S=0;
    for i in range(1):
        S += math.pow(x-y, 2);
    return math.sqrt(S);

def UpdateMean(n, mean, item):
    for i in range(len(mean)):
        m2 = mean[i];
        m2 = (m2 * (n - 1) + item) / float(n);
        mean[i] = round(m2, 3);
    return mean;

def CalculateMeans(k, items, maxIterations=100000):
    cMin=m;
    cMax=M;
    # Initialize means at random points
    means = InitializeMeans(meansl, k, cMin, cMax);
    # Initialize clusters, the array to hold
    # the number of items in a class
    clusterSizes = [0 for i in range(len(means))];
    # An array to hold the cluster an item is in
    belongsTo = [0 for i in range(len(meansl))];
    # Calculate means
    for e in range(maxIterations):
        # If no change of cluster occurs, halt
        noChange = True;
        for i in range(len(meansl)):
            item = meansl[i];
            # Classify item into a cluster and update the
            # corresponding means.
            index = Classify(means, item);
            clusterSizes[index] += 1;
            cSize = clusterSizes[index];
            means[index] = UpdateMean(cSize, means[index], item);
            # Item changed cluster
            if (index != belongsTo[i]):
                noChange = False;
            belongsTo[i] = index;
             # Nothing changed, return
        if (noChange):
            break;
    return means;

def Classify(means, item):
    minimum = sys.maxsize;
    index = -1;
    for i in range(len(means)):
        dis = EuclideanDistance(item, means[i]);
        if (dis < minimum):
            minimum = dis;
            index = i;
    return index;

def FindClusters(means, meansl):
    clusters = [[] for i in range(len(means))]; # Initialize clusters
    for item in meansl:
        index = Classify(means, item);
        clusters[index].append(item);
    return clusters;

means=CalculateMeans(4,meansl)
#means=CalculateMeans(3,meansl)
means.sort()
print("Means = "+str(means))
g=FindClusters(means,meansl)
print("Cluster Means = "+str(g))
## Testing ##
path='images/test/'
files=[]
meansl=[]
m=[]
M=[]
c=0
for r,d,f in os.walk(path):
    for file in f:
        if('.png' in file or '.jpg' in file):
            files.append(os.path.join(r,file))
            c+=1
#print(files)
for input_file in files:
    m = cv2.imread(input_file)
    h,w,bpp = np.shape(m)
    red=[]
    blue=[]
    green=[]
    v=0
    # for py in range(0,h):
    #     for px in range(0,w):
    #         red.append(m[py][px][0])
    #         blue.append(m[py][px][1])
    #         green.append(m[py][px][2])
    # red_max=max(red)
    # blue_max=max(blue)
    # green_max=max(green)
    # red_min=min(red)
    # blue_min=min(blue)
    # green_min=min(green)
    for py in range(0,h):
        for px in range(0,w):
            x=float(m[py][px][0])
            y=float(m[py][px][1])
            z=float(m[py][px][2])
            m[py][px]=x*0.2126+y*0.0722+z*0.7152
    m = (255 / 1) * (m / (255 / 1)) ** 2
    h,w,bpp = np.shape(m)
    n=np.zeros([h,w,bpp], dtype=np.uint8)
    c=0
    sum=0
    for py in range(0,h):
        for px in range(0,w):
            sum+=m[py][px][0]
            c+=1
    for py in range(0,h):
        for px in range(0,w):
            if m[py][px][0]>(sum/c):
                n[py][px]=m[py][px]
                m[py][px]=255
    sum1=0
    z=0
    for py in range(0,h):
        for px in range(0,w):
            if n[py][px][0]!=0:
                sum1+=n[py][px][0]
                z+=1
    mean=sum1/z
    print(input_file+" mean = "+str(mean))
    for i in range(len(means)):
        if mean>means[i]:
            v=i;
    if v!=len(means)-1:
        o1=means[v+1]-mean;
        o2=mean-means[v];
        if o1<o2:
            v=v+1;
        #print(v)
    else:
        v=len(means)-1;
    print('Current weather condition is:')
#     if v==0:
#         print('CLEAR SKY')
#     elif v==1:
#         print('SUNNY')
#     elif v==2:
# #        print('CLOUDY AND SUNNY')
#         print('CLOUDY')
#     elif v==3:
#         print('CLOUDY WITH CHANCES OF RAIN')
    if v==0:
        print('SUNNY')
    elif v==1:
        print('CLOUDY')
    elif v==2:
        print('RAINY')

#sys.stdout.close()
#sys.stdout=stdoutOrigin