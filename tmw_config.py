#!/usr/bin/env python3
# Filename: my_tmw.py
# Author: #cf

##################################################################
###  CONFIG FILE for: Topic Modeling Workflow (tmw)            ###
##################################################################

# Used in the following paper: Christof Schoech, "Topic Modeling French Crime Fiction"
# presented at the Digital Humanities Conference, Sydney, 2015.
# For information on requirements and usage, see the README file.


import tmw
#print(help(topmod))

### Set the general working directory.
wdir = "/home/christof/Dropbox/0-Analysen/2015/hybrid/rf740c/" # end with slash.



################################
###    PREPROCESSING         ###
################################

### tei5reader_fulldocs (standard option)
### Extract selected plain text from XML/TEI files.
inpath = wdir + "master/*.xml"
outfolder = wdir + "1_txt/"
tmw.tei5reader_fulldocs(inpath,outfolder)

### segmenter
### Split entire texts into smaller segments.
inpath = wdir + "1_txt/*.txt"
outfolder = wdir + "2_segs/"
target = 600
sizetolerancefactor = 1.1 # 1 = exact target; >1 = with some tolerance (1.1 = +/- 10%).
preserveparagraphs = True # True|False
tmw.segmenter(inpath, outfolder, target, sizetolerancefactor, preserveparagraphs)

### segments_to_bins: inpath, outfile
### Sort text segments into a fixed number of bins. 
inpath = wdir + "2_segs/*.txt"
outfile = wdir + "segs-and-bins.csv"
##tmw.segments_to_bins(inpath,outfile)

### pretokenize
### Perform some preliminary tokenization.
inpath = wdir + "2_segs/*.txt"
outfolder = wdir + "3_tokens/"
tmw.pretokenize(inpath,outfolder)

### call_treetagger
### Perform lemmatization and POS tagging.
infolder = wdir + "3_tokens/"
outfolder = wdir + "4_tagged/"
tagger = "/home/christof/Programs/TreeTagger/cmd/tree-tagger-french"
tmw.call_treetagger(infolder, outfolder, tagger) 

### make_lemmatext
### Extract selected lemmata from tagged text.
inpath = wdir + "4_tagged/*.trt"
outfolder = wdir + "5_lemmata/"
mode = "frN" # frN=nouns, esN=nouns, frNV=nouns+verbs, frNVAA=nouns+verbs+adj+adverbs 
stoplist_errors = "./extras/fr_stopwords_errors.txt" # in tmw folder
tmw.make_lemmatext(inpath, outfolder, mode, stoplist_errors)



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
tmw.call_mallet_import(mallet_path, infolder, outfolder, outfile, stoplist_project)

### call_mallet_model
### Performs the actual topic modeling. 
mallet_path = "/home/christof/Programs/Mallet/bin/mallet"
inputfile = wdir + "6_mallet/corpus.mallet"
outfolder = wdir + "6_mallet/"
num_topics = "250"
optimize_interval = "100"
num_iterations = "100"
num_top_words = "200"
doc_topics_max = num_topics
num_threads = "4"
tmw.call_mallet_modeling(mallet_path, inputfile, outfolder, num_topics, optimize_interval, num_iterations, num_top_words, doc_topics_max)



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
tmw.create_mastermatrix(corpuspath, outfolder, mastermatrixfile, metadatafile, topics_in_texts, number_of_topics)

### calculate_averageTopicScores
### Based on the mastermatrix, calculates various average topic score datasets.
mastermatrixfile = wdir+"/7_aggregates/mastermatrix.csv"
outfolder = wdir+"7_aggregates/"
# targets: one or several:author|decade|subgenre|author-gender|idno|segmentID|narration
targets = ["author-name", "author-gender", "title", "decade", "subgenre", 
           "idno", "segmentID", "narration", "protagonist-policier"] 
tmw.calculate_averageTopicScores(mastermatrixfile, targets, outfolder)

### save_firstWords
### Saves the first words of each topic to a separate file.
topicWordFile = wdir+"6_mallet/topics-with-words.csv"
outfolder = wdir+"7_aggregates/"
filename = "firstWords.csv"
tmw.save_firstWords(topicWordFile, outfolder, filename)



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
tmw.make_wordle_from_mallet(word_weights_file,topics,words,outfolder,font_path,dpi)

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
targetCategories = ["author-gender", "decade","title"] # one or several: author-name|decade|subgenre|gender|title|
topTopicsShown = 30 
fontscale = 1.0
height = 0 # 0=automatic and variable
dpi = 300
outfolder = wdir+"/8_visuals/topTopics/"
tmw.plot_topTopics(averageDatasets, firstWordsFile, numberOfTopics, targetCategories, topTopicsShown, fontscale, height, dpi, outfolder)

### plot_topItems
### For each topic, creates a barchart with top items from a category. 
averageDatasets = wdir+"/7_aggregates/avg*.csv" 
outfolder = wdir+"/8_visuals/topItems/"
firstWordsFile = wdir+"/7_aggregates/firstWords.csv"
numberOfTopics = 250 # must be actual number of topics modeled. 
targetCategories = ["author-name", "subgenre"] 
# choose one or several from: author-name, decade, subgenre, gender, idno, title, segmentID
topItemsShown = 30 
fontscale = 1.0
height = 0 # 0=automatic and flexible
dpi = 300
tmw.plot_topItems(averageDatasets, outfolder, firstWordsFile, numberOfTopics, targetCategories, topItemsShown, fontscale, height, dpi)

### plot_distinctiveness_heatmap
### For each category, make a heatmap of most distinctive topics. 
averageDatasets = wdir+"/7_aggregates/avg*.csv" 
firstWordsFile = wdir+"/7_aggregates/firstWords.csv"
outfolder = wdir+"/8_visuals/distinctiveness/"
targetCategories = ["subgenre"] 
# choose one or several from: author-name, decade, subgenre, gender, idno, title, segmentID
numberOfTopics = 10 # must be actual number of topics modeled.
topTopicsShown = 5 
fontscale = 1.0
dpi = 300
#tmw.plot_distinctiveness_heatmap(averageDatasets, firstWordsFile, outfolder, targetCategories, numberOfTopics, topTopicsShown, fontscale, dpi)















### make_topic_distribution_plot
### Creates a variety of plots (to be separated out)
aggregates = wdir+"/7_aggregates/avg*.csv" 
outfolder = wdir+"/8_visuals/"
topicwordfile = wdir+"/6_mallet/topics-with-words.csv"
number_of_topics = 250 # must be actual number of topics modeled.
entries_shown = 30 
font_scale = 1.0
height = 0.020 # for barchart; 0=automatic
dpi = 300
mode = "heatmap" # choose one: heatmap|lineplot|areaplot|barchart
topics = ["40","111","155","192"] # for lineplot/areaplot: select one or several topics (list)
target = "author" # for barchart, choose one: author-name|decade|subgenre|gender|idno|segmentID
#tmw.make_topic_distribution_plot(aggregates,outfolder,topicwordfile, number_of_topics,entries_shown,font_scale, height, dpi, mode, topics, target)





################################
###    OTHER/OBSOLETE        ###
################################

### 5c show segment
segmentID = "rf0546§000083"
outfolder = wdir+"/9_sel-segs/"
#tmw.show_segment(wdir,segmentID, outfolder)

### 6a - aggregate_using_bins_and_metadata
corpuspath = wdir + "5_segs"
outfolder = wdir + "7_aggregates/"
topics_in_texts = wdir + "6_mallet/" + "topics-in-texts.csv"
metadatafile = wdir + "metadata.csv"
bindatafile = wdir + "segs-and-bins.csv" # USER: segments or scenes?
target = "subtype" # User: set ranges in tmw.py
#tmw.aggregate_using_bins_and_metadata(corpuspath, outfolder, topics_in_texts, metadatafile, bindatafile, target)

### 6b - create_topicscores_lineplot
inpath = wdir + "7_aggregates/*-lp.csv"  # narrow down as needed
outfolder = wdir + "8_visuals/lineplots/"
topicwordfile = wdir + "6_mallet/topics-with-words.csv"
dpi = 300
height = 0.050
genres = ["detection","noir"] # User: set depending on metadata. Available: noir, detection, criminel, experim., archq., blanche, neopl., susp.
#tmw.create_topicscores_lineplot(inpath,outfolder,topicwordfile,dpi,height,genres)