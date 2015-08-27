#!/usr/bin/env python3
# Filename: my_tmw.py
# Author: #cf
# Version 0.2.0 (2015-08-27)


##################################################################
###  CONFIG FILE for: Topic Modeling Workflow (tmw)            ###
##################################################################

# Used in the following paper: 
# Christof Schoech, "Topic Modeling French Crime Fiction",
# presented at the Digital Humanities Conference, Sydney, 2015.
# For information on requirements and usage, see the README file.

# This config file is structured as follows: 
# 1. Preprocessing Texts
# 2. Topic Modeling
# 3. Posprocessing Data
# 4. Visualization
# 5. Other / Obsolete

import tmw
#print(help(topmod))

### Set the general working directory.
wdir = "/home/.../" # end with slash.

################################
###    PREPROCESSING TEXTS   ###
################################

### tei5reader_fulldocs (standard option)
### Extract selected plain text from XML/TEI files.
inpath = wdir + "master/*.xml"
outfolder = wdir + "1_txt/"
#tmw.tei5reader_fulldocs(inpath,outfolder)

### segmenter
### Split entire texts into smaller segments.
inpath = wdir + "1_txt/*.txt"
outfolder = wdir + "2_segs/"
target = 600
sizetolerancefactor = 1.1 # 1 = exact target; >1 = with some tolerance (1.1 = +/- 10%).
preserveparagraphs = True # True|False
#tmw.segmenter(inpath, outfolder, target, sizetolerancefactor, preserveparagraphs)

### segments_to_bins: inpath, outfile
### Currently not implemented any more / yet. 

### pretokenize
### Perform some preliminary tokenization.
inpath = wdir + "2_segs/*.txt"
outfolder = wdir + "3_tokens/"
substitutionsFile = "./extras/fr_pretokenize_subs.csv"
#tmw.pretokenize(inpath, substitutionsFile, outfolder)

### call_treetagger
### Perform lemmatization and POS tagging.
infolder = wdir + "3_tokens/"
outfolder = wdir + "4_tagged/"
tagger = "/home/christof/Programs/TreeTagger/cmd/tree-tagger-french"
#tmw.call_treetagger(infolder, outfolder, tagger) 

### make_lemmatext
### Extract selected lemmata from tagged text.
inpath = wdir + "4_tagged/*.trt"
outfolder = wdir + "5_lemmata/"
mode = "frN" # frN=nouns, esN=nouns, frNV=nouns+verbs, frNVAA=nouns+verbs+adj+adverbs 
stoplist_errors = "./extras/fr_stopwords_errors.txt" # in tmw folder
#tmw.make_lemmatext(inpath, outfolder, mode, stoplist_errors)



################################
###    TOPIC MODELING        ###
################################

### call_mallet_import
### Imports text data into the Mallet corpus format.
mallet_path = "/home/christof/Programs/Mallet/bin/mallet"
infolder = wdir + "5_lemmata/"
outfolder = wdir + "6_mallet/" 
outfile = outfolder + "corpus.mallet"
stoplist_project = "./extras/fr_stopwords_project.txt" # in tmw folder
#tmw.call_mallet_import(mallet_path, infolder, outfolder, outfile, stoplist_project)

### call_mallet_model
### Performs the actual topic modeling. 
mallet_path = "/home/christof/Programs/Mallet/bin/mallet"
inputfile = wdir + "6_mallet/corpus.mallet"
outfolder = wdir + "6_mallet/"
num_topics = "250"
optimize_interval = "100"
num_iterations = "5000"
num_top_words = "200"
doc_topics_max = num_topics
num_threads = "4"
#tmw.call_mallet_modeling(mallet_path, inputfile, outfolder, num_topics, optimize_interval, num_iterations, num_top_words, doc_topics_max)



################################
###    POSTPROCESSING DATA   ###
################################

### create_mastermatrix
### Creates the mastermatrix with all information in one place.
corpuspath = wdir+"/2_segs/*.txt"
outfolder = wdir+"7_aggregates/"
mastermatrixfile = "mastermatrix.csv"
metadatafile = wdir+"/metadata.csv"
topics_in_texts = wdir+"/6_mallet/topics-in-texts.csv"
number_of_topics = 250
#tmw.create_mastermatrix(corpuspath, outfolder, mastermatrixfile, metadatafile, topics_in_texts, number_of_topics)

### calculate_averageTopicScores
### Based on the mastermatrix, calculates various average topic score datasets.
mastermatrixfile = wdir+"/7_aggregates/mastermatrix.csv"
outfolder = wdir+"7_aggregates/"
# targets: one or several:author|decade|subgenre|author-gender|idno|segmentID|narration
targets = ["author-name", "author-gender", "title", "decade", "subgenre", 
           "idno", "segmentID", "narration", "protagonist-policier"] 
#tmw.calculate_averageTopicScores(mastermatrixfile, targets, outfolder)

### save_firstWords
### Saves the first words of each topic to a separate file.
topicWordFile = wdir+"6_mallet/topics-with-words.csv"
outfolder = wdir+"7_aggregates/"
filename = "firstWords.csv"
#tmw.save_firstWords(topicWordFile, outfolder, filename)



################################
###    VISUALIZATION         ###
################################

### make_wordle_from_mallet
### Creates a wordle for each topic.
word_weights_file = wdir + "6_mallet/" + "word-weights.txt"
topics = 250
words = 40
outfolder = wdir + "8_visuals/wordles/"
font_path = "/home/christof/.fonts/AlegreyaSans-Regular.otf"
dpi = 300
#tmw.make_wordle_from_mallet(word_weights_file,topics,words,outfolder,font_path,dpi)

### crop_images
### Crops the wordle image files, use if needed.
inpath = wdir + "8_visuals/wordles/*.png"
outfolder = wdir + "8_visuals/wordles/"
left = 225 # image start at the left
upper = 210 # image start at the top
right = 2225 # image end on the right
lower = 1310 # image end at the bottom
#tmw.crop_images(inpath, outfolder, left, upper, right, lower)

### plot_topTopics
### For each item from a category, creates a barchart of the top topics.
averageDatasets = wdir+"/7_aggregates/avg*.csv" 
firstWordsFile = wdir+"/7_aggregates/firstWords.csv"
numberOfTopics = 250 # must be actual number of topics modeled.
targetCategories = ["author-name", "author-gender", "decade", "subgenre", "title"] 
# one or several: "author-name", "author-gender", "decade", "subgenre", "title"
topTopicsShown = 30 
fontscale = 1.0
height = 0 # 0=automatic and variable
dpi = 300
outfolder = wdir+"/8_visuals/topTopics/"
#tmw.plot_topTopics(averageDatasets, firstWordsFile, numberOfTopics, targetCategories, topTopicsShown, fontscale, height, dpi, outfolder)

### plot_topItems
### For each topic, creates a barchart with top items from a category. 
averageDatasets = wdir+"/7_aggregates/avg*.csv" 
outfolder = wdir+"/8_visuals/topItems/"
firstWordsFile = wdir+"/7_aggregates/firstWords.csv"
numberOfTopics = 250 # must be actual number of topics modeled. 
targetCategories = ["author-name", "subgenre", "title", "decade", "author-gender"] 
# choose one or several from: author-name, decade, subgenre, gender, idno, title, segmentID
topItemsShown = 30 
fontscale = 0.8
height = 0 # 0=automatic and flexible
dpi = 300
#tmw.plot_topItems(averageDatasets, outfolder, firstWordsFile, numberOfTopics, targetCategories, topItemsShown, fontscale, height, dpi)

### plot_distinctiveness_heatmap
### For each category, make a heatmap of most distinctive topics. 
averageDatasets = wdir+"/7_aggregates/avg*.csv" 
firstWordsFile = wdir+"/7_aggregates/firstWords.csv"
outfolder = wdir+"/8_visuals/distinctiveness/"
targetCategories = ["author-name", "decade", "subgenre", "gender"] 
# one or several: "author-name", "decade", "subgenre", "gender", "idno", "title"
numberOfTopics = 250 # must be actual number of topics modeled.
topTopicsShown = 20 
fontscale = 1.0
dpi = 300
#tmw.plot_distinctiveness_heatmap(averageDatasets, firstWordsFile, outfolder, targetCategories, numberOfTopics, topTopicsShown, fontscale, dpi)

### plot_topicsOverTime
### Creates lineplots or areaplots for topic development over time.
averageDatasets = wdir+"/7_aggregates/avgtopicscores_by-decade.csv" 
firstWordsFile = wdir+"/7_aggregates/firstWords.csv"
outfolder = wdir+"/8_visuals/overTime/"
numberOfTopics = 250 # must be actual number of topics modeled.
fontscale = 1.0
dpi = 300
height = 0 # for lineplot; 0=automatic
mode = "line" # area|line for areaplot or lineplot
topics = ["48","67","199"] # list of one or several topics
#tmw.plot_topicsOverTime(averageDatasets, firstWordsFile, outfolder, numberOfTopics, fontscale, dpi, height, mode, topics)



################################
###    OTHER/OBSOLETE        ###
################################

### 5c show segment
## To read a specific segment, better than looking in the folder.
segmentID = "rf0546§000083"
outfolder = wdir+"/9_sel-segs/"
#tmw.show_segment(wdir,segmentID, outfolder)

### 6b - create_topicscores_lineplot
inpath = wdir + "7_aggregates/*-lp.csv"  # narrow down as needed
outfolder = wdir + "8_visuals/lineplots/"
topicwordfile = wdir + "6_mallet/topics-with-words.csv"
dpi = 300
height = 0.050
genres = ["detection","noir"] # User: set depending on metadata. Available: noir, detection, criminel, experim., archq., blanche, neopl., susp.
#tmw.create_topicscores_lineplot(inpath,outfolder,topicwordfile,dpi,height,genres)
