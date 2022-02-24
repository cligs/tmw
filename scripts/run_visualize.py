#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: my_tmw.py
# Author: #cf
# Version 0.2.0 (2015-08-27)


import visualize
from os.path import join

### Set the general working directory.
wdir = "/home/ulrike/Git/papers/family_resemblance_dsrom19/"
### Set parameters as used in the topic model
NumTopics = 100
NumIterations = 5000
OptimizeIntervals = 100
param_settings = str(NumTopics) + "tp-" + str(NumIterations) + "it-" + str(OptimizeIntervals) + "in"

### make_wordle_from_mallet
### Creates a wordle for each topic.
word_weights_file = join(wdir, "topicmodel", "mallet", "word-weights_" + param_settings + ".csv")
words = 40
outfolder = join(wdir, "topicmodel", "visuals", param_settings, "wordles")
font_path = join(wdir, "extras", "AlegreyaSans-Regular.otf")
dpi = 300
num_topics = NumTopics
TopicRanksFile = join(wdir, "topicmodel", "aggregates", param_settings, "topicRanks.csv")
#visualize.make_wordle_from_mallet(word_weights_file, num_topics, words, TopicRanksFile, outfolder, dpi) # ggf. font_path

### crop_images
### Crops the wordle image files, use if needed.
inpath = join(wdir, "8_visuals", param_settings, "wordles", "*.png")
outfolder = join(wdir, "8_visuals", param_settings, "wordles")
left = 500 # image start at the left
upper = 50 # image start at the top
right = 3400 # image end on the right
lower = 2350 # image end at the bottom
#visualize.crop_images(inpath, outfolder, left, upper, right, lower)

### plot_topTopics
### For each item from a category, creates a barchart of the top topics.
averageDatasets = join(wdir, "features/topicmodel/", "aggregates", param_settings, "avg*.csv") 
firstWordsFile = join(wdir, "features/topicmodel", "aggregates", param_settings, "firstWords.csv")
numberOfTopics = NumTopics # must be actual number of topics modeled.
targetCategories = ["idno"]
# one or several: "author-name", "author-gender", "decade", "subgenre", "title"
topTopicsShown = 30 
fontscale = 1.0
height = 0 # 0=automatic and variable
dpi = 300
outfolder = join(wdir, "features/topicmodel", "visuals", param_settings, "topTopics")
mode = "normalized" # normalized, absolute
#visualize.plot_topTopics(averageDatasets, firstWordsFile, numberOfTopics, targetCategories, mode, topTopicsShown, fontscale, height, dpi, outfolder)

### plot_topItems
### For each topic, creates a barchart with top items from a category. 
averageDatasets = join(wdir, "8_aggregates", param_settings, "avg*.csv") 
outfolder = join(wdir, "9_visuals", param_settings, "topItems")
firstWordsFile = join(wdir, "8_aggregates", param_settings, "firstWords.csv")
numberOfTopics = NumTopics # must be actual number of topics modeled. 
targetCategories = ["author-name", "title", "narrative-perspective", "subgenre", "decade"]
# choose one or several from: author-name, decade, subgenre, gender, idno, title, segmentID
topItemsShown = 20 
fontscale = 0.8
height = 0 # 0=automatic and flexible
dpi = 300
#visualize.plot_topItems(averageDatasets, outfolder, firstWordsFile, numberOfTopics, targetCategories, topItemsShown, fontscale, height, dpi)

### plot_distinctiveness_heatmap
### For each category, make a heatmap of most distinctive topics. 
averageDatasets = join(wdir, "topicmodel/aggregates", param_settings, "avg*.csv") 
firstWordsFile = join(wdir, "topicmodel/aggregates", param_settings, "firstWords.csv")
outfolder = join(wdir, "topicmodel/visuals", param_settings, "distinctiveness")
targetCategories = ["text.genre.subgenre.summary"] 
# one or several: "author-name", "decade", "subgenre", "gender", "idno", "title"
numberOfTopics = NumTopics # must be actual number of topics modeled.
topTopicsShown = 20 
mode = "zscores" # meannorm|mediannorm|zscores|absolute
sorting = "std"
fontscale = 1.0
dpi = 300
#visualize.plot_distinctiveness_heatmap(averageDatasets, firstWordsFile, outfolder, targetCategories, numberOfTopics, topTopicsShown, mode, sorting, fontscale, dpi)

### plot_topicsOverTime
### Creates lineplots or areaplots for topic development over time.
#averageDatasets = wdir+"/7_aggregates/avgtopicscores_by-decade.csv" 
#firstWordsFile = wdir+"/7_aggregates/firstWords.csv"
#outfolder = wdir+"/8_visuals/overTime/"
#numberOfTopics = 250 # must be actual number of topics modeled.
#fontscale = 1.0
#dpi = 300
#height = 0 # for lineplot; 0=automatic
#mode = "line" # area|line for areaplot or lineplot
#topics = ["48","67","199"] # list of one or several topics
#tmw.plot_topicsOverTime(averageDatasets, firstWordsFile, outfolder, numberOfTopics, fontscale, dpi, height, mode, topics)
