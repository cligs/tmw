#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: model.py
# Authors: christofs
# Version 0.3.0 (2016-03-20)

"""
Parameter file for diagnose.py. 
"""

import diagnose
import glob
import os


#################
# Parameters    
#################

### General working directory. End with slash. ###
WorkDir = "/media/christof/data/Dropbox/0-Analysen/2016/gddh-dhq/tc391zz/"
#WorkDir = "/home/christof/Dropbox/0-Analysen/2016/gddh-dhq/tc391y/"

### check_topic_indicators ###
#DiagnoseFile = WorkDir + "7_model/diagnostics_60tp-6000it-0300in.xml"
DiagnoseFilePath = WorkDir + "7_model/diagnostics_*.xml"
IndicatorType = "word-length" #coherence|eff_num_words|corpus_dist|word-length
GraphFileIS = WorkDir + "8_diagnostics/topic-indicators_"+IndicatorType+".svg"

### topicscore_distributions ###
# File with topic probability scores in the model
TopicScoreFilesPath = WorkDir + "7_model/topics-with-words_60tp*.csv"
# Graph file for results from topic score distribution
GraphFileTSD = WorkDir + "8_diagnostics/topic-score-distr.svg"

### count_words
TopicsWithWordsFilePath = WorkDir + "7_model/topics-with-words_60tp*.csv"

#WordWeightFilesPath = WorkDir + "6_mallet-100tp_betafix//word-weights*.txt"
#GraphFileWWD = WorkDir + "6_mallet-100tp_betafix/word-weight-distr.svg"


#################
# Functions    
#################

# Run functions
#diagnose.check_topicindicators(DiagnoseFilePath, IndicatorType, GraphFileIS)
diagnose.topicscore_distributions(TopicScoreFilesPath, GraphFileTSD)
#diagnose.count_words(TopicsWithWordsFilePath)



