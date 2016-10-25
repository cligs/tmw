#!/usr/bin/env python3
# Filename: run_model.py
# Author: #cf
# Version 0.1.0 (2016-10-17)

"""
Parameter file for model.py. 
"""

import model


### Set the general working directory. End with slash.
WorkDir = "/media/christof/data/Dropbox/0-Analysen/2016/gddh-dhq/tc391zz/"


### Shared parameters
MalletPath = "/media/christof/data/repos/other/Mallet/bin/mallet"
TextFolder = WorkDir + "5_lemmata/"
MalletFolder = WorkDir + "6_mallet/" 
CorpusFile = MalletFolder + "tc.mallet"


### Import parameters (call_mallet_import)
StoplistProject = "./extras/fr-lemma_stopwords-project.txt"
model.call_mallet_import(MalletPath, TextFolder, 
                         MalletFolder, CorpusFile, StoplistProject)


### Modeling parameters (call_mallet_model)
NumTopics = [50, 60, 70, 80, 90, 100]
NumIterations = [6000]
OptimizeIntervals = [50, 100, 300, 500, 1000, 2000, 3000, 7000]
NumTopWords = 50
NumThreads = 4
ModelFolder = WorkDir + "7_model/"

model.call_mallet_modeling(MalletPath, CorpusFile, ModelFolder, 
                           NumTopics, NumIterations, OptimizeIntervals, 
                           NumTopWords, NumThreads)

