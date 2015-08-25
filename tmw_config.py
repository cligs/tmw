#!/usr/bin/env python3
# Filename: my_tmw.py

"""
# Python module for a Topic Modeling Workflow 
"""
 
# Used in the following paper: Christof Schoech, "Topic Modeling French Crime Fiction"
# presented at the Digital Humanities Conference, Sydney, 2015.
# For information on requirements and usage, see the README file.


import tmw
#print(help(topmod))

### Set the general working directory.
wdir = "/home/christof/Dropbox/0-Analysen/2015/hybrid/rf740/" # end with slash.

### 1a - tei5reader_fulldocs (standard option)
inpath = wdir + "master/*.xml"
outfolder = wdir + "1_txt/"
tmw.tei5reader_fulldocs(inpath,outfolder)

### 1b - segmenter
inpath = wdir + "1_txt/*.txt"
outfolder = wdir + "2_segs/"
target = 600
sizetolerancefactor = 1.1 # 1 = exact target; >1 = with some tolerance (1.1 = +/- 10%).
preserveparagraphs = True # True|False
tmw.segmenter(inpath, outfolder, target, sizetolerancefactor, preserveparagraphs)

### 1c - segments_to_bins: inpath, outfile
inpath = wdir + "2_segs/*.txt"
outfile = wdir + "segs-and-bins.csv"
##tmw.segments_to_bins(inpath,outfile)



### 2a - pretokenize
inpath = wdir + "2_segs/*.txt"
outfolder = wdir + "3_tokens/"
tmw.pretokenize(inpath,outfolder)

### 2b - call_treetagger
infolder = wdir + "3_tokens/"
outfolder = wdir + "4_tagged/"
tagger = "/home/christof/Programs/TreeTagger/cmd/tree-tagger-french"
tmw.call_treetagger(infolder, outfolder, tagger) 

### 2c - make_lemmatext
inpath = wdir + "4_tagged/*.trt"
outfolder = wdir + "5_lemmata/"
mode = "N" # N=nouns, NV=nouns+verbs, NVAA=nouns+verbs+adj+adverbs
stoplist = ["<unknown>", "unknown"]
tmw.make_lemmatext(inpath, outfolder, mode, stoplist)



### 3a - call_mallet_import
mallet_path = "/home/christof/Programs/Mallet/bin/mallet"
infolder = wdir + "5_lemmata/"
outfolder = wdir + "6_mallet/" 
outfile = outfolder + "corpus.mallet"
stoplist = "./extras/stopwords_fr.txt" # in tmw folder
tmw.call_mallet_import(mallet_path, infolder, outfolder, outfile, stoplist)


### 3b - call_mallet_model
mallet_path = "/home/christof/Programs/Mallet/bin/mallet"
inputfile = wdir + "6_mallet/corpus.mallet"
outfolder = wdir + "6_mallet/"
num_topics = "250"
optimize_interval = "100"
num_iterations = "10000"
num_top_words = "200"
doc_topics_max = num_topics
num_threads = "4"
tmw.call_mallet_modeling(mallet_path, inputfile, outfolder, num_topics, optimize_interval, num_iterations, num_top_words, doc_topics_max)



### 4 - make_wordle_from_mallet
word_weights_file = wdir + "6_mallet/" + "word-weights.txt"
topics = 250
words = 40
outfolder = wdir + "8_visuals/wordles/"
font_path = "/home/christof/.fonts/AlegreyaSans-Regular.otf"
dpi = 300
tmw.make_wordle_from_mallet(word_weights_file,topics,words,outfolder,font_path,dpi)



### 5a - average_topicscores
corpuspath = wdir+"/2_segs/*.txt"
mastermatrixfile = wdir+"/7_aggregates/mastermatrix.csv"
metadatafile = wdir+"/metadata.csv"
topics_in_texts = wdir+"/6_mallet/topics-in-texts.csv"
#targets = ["author"] 
targets = ["author-name","decade","subgenre","author-gender","idno","segmentID", "narration", "protagonist-policier"] 
# targets: one or several:author|decade|subgenre|author-gender|idno|segmentID|narration
mode = "create" # load|create mastermatrix
number_of_topics = 250
outfolder = wdir+"7_aggregates/"
#tmw.average_topicscores(corpuspath, mastermatrixfile, metadatafile, topics_in_texts, targets, mode, number_of_topics, outfolder)



### 5b make_topic_distribution_plot
aggregates = wdir+"/7_aggregates/avg*.csv" 
outfolder = wdir+"/8_visuals/"
topicwordfile = wdir+"/6_mallet/topics-with-words.csv"
number_of_topics = 250 # must be actual number of topics modeled.
entries_shown = 30 
font_scale = 1.0
height = 0 # for barchart; 0=automatic
dpi = 300
mode = "barchart" # choose one: heatmap|lineplot|areaplot|barchart
topics = ["40","111","155","192"] # for lineplot/areaplot: select one or several topics (list)
target = "author-name" # for barchart, choose one: author-name|decade|subgenre|gender|idno|segmentID
#tmw.make_topic_distribution_plot(aggregates,outfolder,topicwordfile, number_of_topics,entries_shown,font_scale, height, dpi, mode, topics, target)


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





