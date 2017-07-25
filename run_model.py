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
wdir = "/home/ulrike/Dokumente/GS/Veranstaltungen/WS16-17_Praxisworkshop"


### Shared parameters
MalletPath = "/home/ulrike/Programme/mallet-2.0.8RC3/bin/mallet"
TextFolder = join(wdir, "5_lemmata")
MalletFolder = join(wdir, "6_mallet") 
CorpusFile = join(MalletFolder, "tc.mallet")


### Import parameters (call_mallet_import)
StoplistProject = join(wdir, "extras", "stopwords_es.txt")
#model.call_mallet_import(MalletPath, TextFolder, MalletFolder, CorpusFile, StoplistProject)


### Modeling parameters (call_mallet_model)
NumTopics = [50]
NumIterations = [1000]
OptimizeIntervals = [50]
NumTopWords = 50
NumThreads = 4
ModelFolder = join(wdir, "7_model")

#model.call_mallet_modeling(MalletPath, CorpusFile, ModelFolder, NumTopics, NumIterations, OptimizeIntervals, NumTopWords, NumThreads)
