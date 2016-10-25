#!/usr/bin/env python3
# Filename: my_tmw.py
# Author: #cf
# Version 0.2.0 (2015-08-27)


import postprocess

### Set the general working directory.
wdir = "/media/christof/data/Dropbox/0-Analysen/2016/gddh-dhq/tc391zz/" # end with slash.


### create_mastermatrix
### Creates the mastermatrix with all information in one place.
corpuspath = wdir+"/2_segs/*.txt"
outfolder = wdir+"9_aggregates/060tp-0300in/"
mastermatrixfile = "mastermatrix.csv"
metadatafile = wdir+"metadata.csv"
topics_in_texts = wdir+"7_model/topics-in-texts_060tp-6000it-0300in.csv"
number_of_topics = 60
useBins = True
binDataFile = wdir + "3_bins/segs-and-bins.csv"
version  = "208+"
postprocess.create_mastermatrix(corpuspath, outfolder, mastermatrixfile, metadatafile, topics_in_texts, number_of_topics, useBins, binDataFile, version)

### calculate_averageTopicScores
### Based on the mastermatrix, calculates various average topic score datasets.
mastermatrixfile = wdir+"/9_aggregates/060tp-0300in/mastermatrix.csv"
outfolder = wdir+"9_aggregates/060tp-0300in/"
# targets: one or several:author|decade|subgenre|author-gender|idno|segmentID|narration
targets = ["idno", "tc_subgenre", "tc_author-short"] 
postprocess.calculate_averageTopicScores(mastermatrixfile, targets, outfolder)

### save_firstWords
### Saves the first words of each topic to a separate file.
topicWordFile = wdir+"7_model/topics-with-words_060tp-6000it-0300in.csv"
outfolder = wdir+"9_aggregates/060tp-0300in/"
filename = "firstWords.csv"
postprocess.save_firstWords(topicWordFile, outfolder, filename)

### Save topic ranks
topicWordFile = wdir+"7_model/topics-with-words_060tp-6000it-0300in.csv"
outfolder = wdir+"9_aggregates/060tp-0300in/"
filename = "topicRanks.csv"
postprocess.save_topicRanks(topicWordFile, outfolder, filename)

### Average topic scores for two criteria (binID + subgenre)
mastermatrixfile = wdir+"/9_aggregates/060tp-0300in/mastermatrix.csv"
targets = ["binID", "tc_subgenre"]
outfolder = wdir+"9_aggregates/060tp-0300in/"
postprocess.calculate_complexAverageTopicScores(mastermatrixfile, targets, outfolder)
