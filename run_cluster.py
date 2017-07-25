#!/usr/bin/env python3
# Filename: run_topicpca.py
# Author: #cf

"""
Set of functions to perform clustering on data. 
Built for topic probabilities or word frequencies as input. 
Performs Principal Component Analysis or distance-based clustering.
### This is the parameter file. ###
"""

import cluster

########################
### Working Directory
########################
WorkDir = "/media/christof/data/Dropbox/0-Analysen/2016/gddh-dhq/tc391zz/"



########################
### topic clustering
########################

MastermatrixFile = WorkDir + "9_aggregates/060tp-0300in/mastermatrix.csv"
MetadataFile = WorkDir + "metadata.csv"
GraphFolder = WorkDir + "9_visuals/060tp-0300in/topiccluster/"
Method = "ward" # ward|average
Metric = "euclidean"  # euclidean|cosine
DisplayLevels = 20
#cluster.topiccluster(MastermatrixFile, 
#                     MetadataFile,
#                     Method,
#                     Metric,
#                     GraphFolder,
#                     DisplayLevels)



########################
### word clustering
########################

WordFreqsFile = WorkDir + "doctermmatrix.csv"
MetadataFile = WorkDir + "metadata.csv"
AllMFW = [3000]
GraphFolder = WorkDir + "9_visuals/060tp-0300in/wordcluster/"
Method = "average" # ward|average
Metric = "cosine"  # euclidean|cosine
DisplayLevels = 20
#cluster.wordcluster(WordFreqsFile,
#                    AllMFW,
#                     MetadataFile,
#                     Method,
#                     Metric,
#                     GraphFolder,
#                     DisplayLevels)





########################
### topicpca
########################
MastermatrixFile = WorkDir + "9_aggregates/060tp-0300in/mastermatrix.csv"
MetadataFile = WorkDir + "metadata.csv"
GraphFolder = WorkDir + "9_visuals/060tp-0300in/topicpca/"
cluster.topicpca(MastermatrixFile,
                 MetadataFile,
                 GraphFolder)


########################
### wordpca
########################### 
WordfreqsFile = WorkDir + "doctermmatrix.csv"
MetadataFile = WorkDir + "metadata.csv"
GraphFolder = WorkDir + "9_visuals/060tp-0300in/wordpca/"
AllMFW = [100, 300, 500, 1000, 2000, 3000, 4000, 5000]
#cluster.wordpca(WordfreqsFile,
#              MetadataFile,
#              GraphFolder,
#              AllMFW)






