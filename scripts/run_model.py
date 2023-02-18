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
wdir = "/home/ulrike/Dokumente/Konferenzen/2022_ML-Ed/johnson-topics"


### Shared parameters
MalletPath = "/home/ulrike/Programme/mallet-2.0.8RC3/bin/mallet"
#MalletPath = "/media/christof/data/repos/other/Mallet/bin/mallet" # doesn't work right now.
TextFolder = join(wdir, "data/letters-leipzig_lemmata_N")
MalletFolder = join(wdir, "tm/mallet_N") 
CorpusFile = join(MalletFolder, "ub.mallet")


### Import parameters (call_mallet_import)
StoplistProject = join(wdir, "data/stopwords.txt")
model.call_mallet_import(MalletPath, TextFolder, MalletFolder, CorpusFile, StoplistProject)


### Modeling parameters (call_mallet_model)
NumTopics = [10,15]
NumIterations = [5000]
OptimizeIntervals = [None]
NumRepetitions = 1
NumTopWords = 50
NumThreads = 4
ModelFolder = join(wdir, "tm/mallet")

model.call_mallet_modeling(MalletPath, CorpusFile, ModelFolder, NumTopics, NumIterations, OptimizeIntervals, NumRepetitions, NumTopWords, NumThreads)
