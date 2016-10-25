#!/usr/bin/env python3
# Filename: my_tmw.py
# Author: #cf
# Version 0.2.0 (2015-08-27)

import prepare

### Set the general working directory.
wdir = "/media/christof/data/Dropbox/0-Analysen/2016/gddh-dhq/tc391zz/" # end with slash.


### segmenter
### Split entire texts into smaller segments.
inpath = wdir + "1_txt/*.txt"
outfolder = wdir + "2_segs/"
target = 1200
sizetolerancefactor = 1.1 # 1 = exact target; >1 = with some tolerance (1.1 = +/- 10%).
preserveparagraphs = True # True|False
#prepare.segmenter(inpath, outfolder, target, sizetolerancefactor, preserveparagraphs)

### segments_to_bins: inpath, outfile
inpath = wdir + "2_segs/*.txt"
outfolder = wdir + "3_bins/" 
binsnb = 5
prepare.segments_to_bins(inpath, outfolder, binsnb)


### call_treetagger
### Perform lemmatization and POS tagging.
infolder = wdir + "2_segs/"
outfolder = wdir + "4_tagged/"
tagger = "/home/christof/Programs/TreeTagger/cmd/tree-tagger-french"
#prepare.call_treetagger(infolder, outfolder, tagger) 

### make_lemmatext
### Extract selected lemmata from tagged text.
inpath = wdir + "4_tagged/*.trt"
outfolder = wdir + "5_lemmata/"
mode = "frNVA" # frN=nouns, esN=nouns, frNV=nouns+verbs, frNVAA=nouns+verbs+adj+adverbs 
stoplist_errors = "./extras/fr-lemma_stopwords-errors.txt" # in tmw folder
#prepare.make_lemmatext(inpath, outfolder, mode, stoplist_errors)


