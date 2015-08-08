#!/usr/bin/env python3
# Filename: my_tmw.py

"""
# Python module for a Topic Modeling Workflow 
"""
 
# Used in the following paper: Christof Schöch, "Topic Modeling French Crime Fiction"
# presented at the Digital Humanities Conference, Sydney, 2015.
# For information on requirements and usage, see the README file.


import tmw
#print(help(topmod))

### Set the general working directory.
wdir = "/home/christof/Dropbox/0-Analysen/2015/hybrid/rf650/" # end with slash.

### 1a - tei5reader_fulldocs (standard option)
inpath = wdir + "master/*.xml"
outfolder = wdir + "1_txt/"
#tmw.tei5reader_fulldocs(inpath,outfolder)

### 1b - segmenter
inpath = wdir + "1_txt/*.txt"
outpath = wdir + "2_segs/"
segment_length = 1000
#tmw.segmenter(inpath,outpath,segment_length)

### 1c - segments_to_bins: inpath, outfile
inpath = wdir + "2_segs/*.txt"
outfile = wdir + "segs-and-bins.csv"
#tmw.segments_to_bins(inpath,outfile)



### 2a - pretokenize
inpath = wdir + "2_segs/*.txt"
outfolder = wdir + "3_tokens/"
#tmw.pretokenize(inpath,outfolder)

### 2b - call_treetagger
infolder = wdir + "3_tokens/"
outfolder = wdir + "4_tagged/"
tagger = "/home/christof/Programs/TreeTagger/cmd/tree-tagger-french"
#tmw.call_treetagger(infolder, outfolder, tagger) 

### 2c - make_lemmatext
inpath = wdir + "4_tagged/*.trt"
outfolder = wdir + "5_lemmata/"
#tmw.make_lemmatext(inpath,outfolder)



### 3a - call_mallet_import
mallet_path = "/home/christof/Programs/Mallet/bin/mallet"
infolder = wdir + "5_lemmata/"
outfolder = wdir + "6_mallet/" 
outfile = outfolder + "corpus.mallet"
stoplist = "./extras/stopwords_fr.txt" # put in tmw folder!
#tmw.call_mallet_import(mallet_path, infolder, outfolder, outfile, stoplist)

### 3b - call_mallet_model
mallet_path = "/home/christof/Programs/Mallet/bin/mallet"
inputfile = wdir + "6_mallet/corpus.mallet"
outfolder = wdir + "6_mallet/"
num_topics = "200"
optimize_interval = "100"
num_iterations = "5000"
num_top_words = "100"
doc_topics_max = "100"
num_threads = "4"
#tmw.call_mallet_modeling(mallet_path, inputfile, outfolder, num_topics, optimize_interval, num_iterations, num_top_words, doc_topics_max)



### 4 - make_wordle_from_mallet
word_weights_file = wdir + "6_mallet/" + "word-weights.txt"
topics = 100
words = 40
outfolder = wdir + "8_visuals/wordles/"
font_path = "/home/christof/.fonts/AlegreyaSans-Regular.otf"
dpi = 300
tmw.make_wordle_from_mallet(word_weights_file,topics,words,outfolder,font_path,dpi)



### 5a - aggregate_using_metadata
corpuspath = wdir + "5_segs"
outfolder = wdir + "7_aggregates/"
topics_in_texts = wdir + "6_mallet/topics-in-texts.csv"
metadatafile = wdir + "metadata.csv"
#target = "subgenre" # USER: set depending on available metadata
#targets = ["idno","author","decade","subgenre","label","narr"] # USER: set depending on available metadata
targets = ["decade"] # USER: set depending on available metadata
#tmw.aggregate_using_metadata(corpuspath,outfolder,topics_in_texts,metadatafile,targets)

### 5b - create_topicscores_heatmap
inpath = wdir + "7_aggregates/*DECADE-hm.csv"
outfolder = wdir + "8_visuals/heatmaps/"
rows_shown = 20
font_scale = 1.0
dpi = 300
#tmw.create_topicscores_heatmap(inpath,outfolder,rows_shown,font_scale,dpi)

### 5c make_topic_distribution_heatmap
aggregates = wdir + "/7_aggregates/topics_by_SUBGENRE*.csv"
outfolder = "/home/christof/Desktop/" 
topicwordfile = wdir+"/6_mallet/topics-with-words.csv"
rows_shown = 20
font_scale = 1.0
dpi = 300
#tmw.make_topic_distribution_heatmap(aggregates,outfolder,topicwordfile,rows_shown,font_scale,dpi)




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





