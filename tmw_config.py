#!/usr/bin/env python3
# Filename: my_tmw.py
# Author: #cf

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
# 4. Basic Visualizations
# 5. Advanced Visualizations
# 6. Other / Obsolete

import tmw
#print(help(topmod))

### Set the general working directory.
wdir = "/home/christof/Dropbox/0-Analysen/2015/hybrid/rf10/" # end with slash.

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
### target: The desired length of each text segment in words. 
### sizetolerancefactor: 1=exact target; >1 = some tolerance, e.g. 1.1= +/-10%.
### preserveparagraphs: True|False, whether \n from input are kept in output.
inpath = wdir + "1_txt/*.txt"
outfolder = wdir + "2_segs/"
target = 2000
sizetolerancefactor = 1.1
preserveparagraphs = True
tmw.segmenter(inpath, outfolder, target, sizetolerancefactor, preserveparagraphs)

### segments_to_bins
inpath = wdir + "2_segs/*.txt"
outfile = wdir + "segs-and-bins.csv"
binsnb = 5 # number of bins
tmw.segments_to_bins(inpath,outfile, binsnb)

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
stoplist_errors = wdir+"extras/fr_stopwords_errors.txt" # in tmw folder
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
stoplist_project = wdir+"extras/fr_stopwords_project.txt" # in tmw folder
#tmw.call_mallet_import(mallet_path, infolder, outfolder, outfile, stoplist_project)

### call_mallet_model
### Performs the actual topic modeling. 
### num_topics: Number of different topics the model should find.
### optimize_interval: interval between hypermarameter optimization.
### num_iterations: How many times the model is improved. 
### num_top_words: Number of words to save and display for each topic.
### num_threads: Number of parallel processing threads to use. 
mallet_path = "/home/christof/Programs/Mallet/bin/mallet"
inputfile = wdir + "6_mallet/corpus.mallet"
outfolder = wdir + "6_mallet/"
num_topics = "250"
optimize_interval = "100"
num_iterations = "1000"
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
### targets: one or several:author|decade|subgenre|author-gender|idno|segmentID|narration
mastermatrixfile = wdir+"/7_aggregates/mastermatrix.csv"
outfolder = wdir+"7_aggregates/"
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
###  BASIC VISUALIZATION     ###
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
### targetCategories: one or several: "author-name", "author-gender", "decade", "subgenre", "title"
### numberOfTopics: Must be the actual number of topics modeled before.
averageDatasets = wdir+"/7_aggregates/avg*.csv" 
firstWordsFile = wdir+"/7_aggregates/firstWords.csv"
targetCategories = ["author-name", "author-gender", "decade", "subgenre", "title"] 
topTopicsShown = 30 
numberOfTopics = 250 
fontscale = 1.0
height = 0 # 0=automatic and variable
dpi = 300
outfolder = wdir+"/8_visuals/topTopics/"
#tmw.plot_topTopics(averageDatasets, firstWordsFile, numberOfTopics, targetCategories, topTopicsShown, fontscale, height, dpi, outfolder)

### plot_topItems ###
### For each topic, creates a barchart with top items from a category. 
### targetCategories: one or several from the following list:
### "author-name", "decade", "subgenre", "gender", "idno", "title", "segmentID"
averageDatasets = wdir+"/7_aggregates/avg*.csv" 
outfolder = wdir+"/8_visuals/topItems/"
firstWordsFile = wdir+"/7_aggregates/firstWords.csv"
numberOfTopics = 250 # must be actual number of topics modeled. 
targetCategories = ["author-name", "subgenre", "title", "decade", "author-gender", "segmentID"] 
topItemsShown = 30 
fontscale = 0.8
height = 0 # 0=automatic and flexible
dpi = 300
#tmw.plot_topItems(averageDatasets, outfolder, firstWordsFile, numberOfTopics, targetCategories, topItemsShown, fontscale, height, dpi)



################################
###  ADVANCED VISUALIZATION  ###
################################

### plot_distinctiveness_heatmap ###
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

### plot_topicsOverTime ###
### Creates lineplots or areaplots for topic development over time.
averageDatasets = wdir+"/7_aggregates/avgtopicscores_by-decade.csv" 
firstWordsFile = wdir+"/7_aggregates/firstWords.csv"
outfolder = wdir+"/8_visuals/overTime/"
numberOfTopics = 250 # must be actual number of topics modeled.
fontscale = 1.0
dpi = 300
height = 0 # for lineplot; 0=automatic
mode = "line" # area|line for areaplot or lineplot
topics = ["25","60"] # list of one or several topics
#tmw.plot_topicsOverTime(averageDatasets, firstWordsFile, outfolder, numberOfTopics, fontscale, dpi, height, mode, topics)

### topicClustering ###
# This function will create a dendrogram grouping topics based on their word weight similarity.
# Parameters 
# wordsPerTopic: Number of top words for each topic to take into account for similarity measure.
# method: The clustering method used to build the dendrogram. 
#  Options: ward|single|complete|average|weighted|centroid|median
#  See http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.linkage.html 
# metric: The distance measure used to build the distance matrix.
#  Options: euclidean|minkowski|cityblock|seuclidean|sqeuclidean|cosine|correlation|hamming|jaccard etc.
#  See: http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
#  Interesting combination: *weighted+cosine  
wordWeightsFile = wdir + "6_mallet/" + "word-weights.txt"
outfolder = wdir + "8_visuals/clustering/"
topicsToUse = 250 # = all topics modeled
wordsPerTopic = 50
methods=["weighted"] # list
metrics=["cosine"] # list
#tmw.topicClustering(wordWeightsFile, wordsPerTopic, outfolder, methods, metrics, topicsToUse)

### itemClustering ###
# This function creates a dendrogram of items in a category (authors, titles).
# The clustering is based on the topic scores of the items. 
# Input: the average topic score file for the category of interest. 
# Parameters
# figsize: The size of the resulting figure in inches, width x height.
# sortingCriterium: Topics to be used are sorted by this criterium (descending)
# topicsPerItem: Number of top topics to be used as the basis for clustering.
# targetCategories: Things like author, title, year, depending on available data.
# method: The clustering method used to build the dendrogram. 
#  Options: ward|single|complete|average|weighted|centroid|median
#  See http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.linkage.html 
# metric: The distance measure used to build the distance matrix.
#  Options: euclidean|minkowski|cityblock|seuclidean|sqeuclidean|cosine|correlation|hamming|jaccard etc.
#  See: http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
#  Interesting combination: *weighted+cosine  
averageDatasets = wdir+"/7_aggregates/avg*title.csv" 
figsize = (10,80) # width,height
outfolder = wdir + "8_visuals/clustering/"
topicsPerItem = 250
sortingCriterium = "std" # std|mean
targetCategories = ["title"] # list
methods=["weighted"] # list
metrics=["cosine"] # list
#tmw.itemClustering(averageDatasets, figsize, outfolder, topicsPerItem, targetCategories, methods, metrics, sortingCriterium)


### itemPCA ###
averageDatasets = wdir+"/7_aggregates/avg*.csv" 
figsize = (10,10) # width,height
outfolder = wdir + "8_visuals/clustering/"
topicsPerItem = 250
sortingCriterium = "std" # std|mean
targetCategories = ["subgenre"] # list
methods=["weighted"] # list
metrics=["cosine"] # list
#tmw.itemPCA(averageDatasets, targetCategories, topicsPerItem, sortingCriterium, figsize, outfolder)



################################
###    OTHER/OBSOLETE        ###
################################

### 5c show segment
## To read a specific segment, better than looking in the folder.
segmentID = "rf0166§0118"
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
