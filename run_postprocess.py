#!/usr/bin/env python3
# Filename: my_tmw.py
# Author: #cf
# Version 0.2.0 (2015-08-27)


import postprocess

### Set the general working directory.
wdir = "/home/ulrike/Dokumente/GS/Veranstaltungen/TM-Workshop-CCeH/Daten-pt/N/" # end with slash.
### Set parameters as used in the topic model
NumTopics = 30
NumIterations = 1000
OptimizeIntervals = 100
param_settings = str(NumTopics) + "tp-" + str(NumIterations) + "it-" + str(OptimizeIntervals) + "in"

### create_mastermatrix
### Creates the mastermatrix with all information in one place.
corpuspath = wdir+"/2_segs/*.txt"
outfolder = wdir+"8_aggregates/" + param_settings + "/"
mastermatrixfile = "mastermatrix.csv"
metadatafile = wdir+"metadata.csv"
topics_in_texts = wdir+"7_model/topics-in-texts_" + param_settings + ".csv"
number_of_topics = NumTopics
useBins = True
binDataFile = wdir + "3_bins/segs-and-bins.csv"
version  = "208+" # which MALLET version is in use?
#postprocess.create_mastermatrix(corpuspath, outfolder, mastermatrixfile, metadatafile, topics_in_texts, number_of_topics, useBins, binDataFile, version)

### calculate_averageTopicScores
### Based on the mastermatrix, calculates various average topic score datasets.
mastermatrixfile = wdir+"/8_aggregates/" + param_settings + "/mastermatrix.csv"
outfolder = wdir+"8_aggregates/" + param_settings + "/"
# targets: one or several:author|decade|subgenre|author-gender|idno|segmentID|narration|narrative-perspective (according to available metadata)
targets = ["idno", "author-name", "title", "narrative-perspective", "subgenre", "decade", "segmentID"] 
#postprocess.calculate_averageTopicScores(mastermatrixfile, targets, outfolder)

### save_firstWords
### Saves the first words of each topic to a separate file.
topicWordFile = wdir+"7_model/topics-with-words_" + param_settings + ".csv"
outfolder = wdir+"8_aggregates/" + param_settings + "/"
filename = "firstWords.csv"
#postprocess.save_firstWords(topicWordFile, outfolder, filename)

### Save topic ranks
topicWordFile = wdir+"7_model/topics-with-words_" + param_settings + ".csv"
outfolder = wdir+"8_aggregates/" + param_settings + "/"
filename = "topicRanks.csv"
#postprocess.save_topicRanks(topicWordFile, outfolder, filename)

### Average topic scores for two criteria (binID + subgenre)
mastermatrixfile = wdir+"/8_aggregates/" + param_settings + "/mastermatrix.csv"
targets = ["binID", "subgenre"]
outfolder = wdir+"8_aggregates/" + param_settings + "/"
postprocess.calculate_complexAverageTopicScores(mastermatrixfile, targets, outfolder)
