#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: run_model.py
# Author: #cf
# Version 0.1.0 (2016-10-17)

"""
Parameter file for model.py. 
"""

import model
from os.path import join


### Set the general working directory.
wdir = "/media/christof/data/repos/cligs/tmw/"


### Shared parameters
MalletPath = "/home/ulrike/Programme/mallet-2.0.8RC3/bin/mallet"
#MalletPath = "/media/christof/data/repos/other/Mallet/bin/mallet" # doesn't work right now.
TextFolder = join(wdir, "data", "5_lemmata")
MalletFolder = join(wdir, "data", "6_mallet") 
CorpusFile = join(MalletFolder, "tc.mallet")


### Import parameters (call_mallet_import)
StoplistProject = join(wdir, "data", "en_stopwords.txt")
model.call_mallet_import(MalletPath, TextFolder, MalletFolder, CorpusFile, StoplistProject)


### Modeling parameters (call_mallet_model)
NumTopics = [20]
NumIterations = [1000]
OptimizeIntervals = [100]
NumTopWords = 10
NumThreads = 4
ModelFolder = join(wdir, "model", "mallet")

model.call_mallet_modeling(MalletPath, CorpusFile, ModelFolder, NumTopics, NumIterations, OptimizeIntervals, NumTopWords, NumThreads)
