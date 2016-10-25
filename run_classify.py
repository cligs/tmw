#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: run_classify.py
# Author: #cf (2016)

import classify

################################
# Parameters
################################


WorkDir = "/media/christof/data/Dropbox/0-Analysen/2016/gddh-dhq/tc391zz/"
TopicScoresFilePath = WorkDir + "7_model/topics-in-texts_*.csv"
WordFreqsFile = WorkDir + "doctermmatrix.csv"
MetadataFile = WorkDir + "metadata.csv"
ClassifierTypes = ["SVM", "KNN", "TRE", "SGD"] # SVM|KNN|TRE|SGD
Target = "tc_subgenre"
InputData = ["words", "topics"]
WordResultsFile = WorkDir + "8_diagnostics/classify_word-results.csv"
TopicResultsFile = WorkDir + "8_diagnostics/classify_topic-results.csv"
GraphFolder = WorkDir + "8_diagnostics/"


#classify.classify(TopicScoresFilePath, 
#                  WordFreqsFile,
#                  MetadataFile, 
#                  Target,
#                  InputData,
#                  ClassifierTypes,
#                  WordResultsFile,
#                  TopicResultsFile)


classify.plot(WordResultsFile,
              TopicResultsFile,
              GraphFolder)