#!/usr/bin/env python3
# Filename: run_model.py
# Author: #cf
# Version 0.1.0 (2016-10-17)

"""
Parameter file for model.py. 
"""

import model


### Set the general working directory. End with slash.
wdir = "/home/ulrike/Dokumente/GS/Veranstaltungen/TM-Workshop-CCeH/Daten-pt/N/"


### Shared parameters
MalletPath = "/home/ulrike/Programme/mallet-2.0.8RC3/bin/mallet"
TextFolder = wdir + "5_lemmata/"
MalletFolder = wdir + "6_mallet/" 
CorpusFile = MalletFolder + "tc.mallet"


### Import parameters (call_mallet_import)
StoplistProject = wdir + "/extras/stopwords_pt.txt"
#model.call_mallet_import(MalletPath, TextFolder, MalletFolder, CorpusFile, StoplistProject)


### Modeling parameters (call_mallet_model)
NumTopics = [30]
NumIterations = [1000]
OptimizeIntervals = [100]
NumTopWords = 50
NumThreads = 4
ModelFolder = wdir + "7_model/"

#model.call_mallet_modeling(MalletPath, CorpusFile, ModelFolder, NumTopics, NumIterations, OptimizeIntervals, NumTopWords, NumThreads)
