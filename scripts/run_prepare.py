#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: my_tmw.py
# Author: #cf
# Version 0.2.0 (2015-08-27)

import prepare
from os.path import join

### Set the general working directory.
wdir = "/home/ulrike/Git/papers/family_resemblance_dsrom19/"


### segmenter
### Split entire texts into smaller segments.
inpath = join(wdir, "topicmodel/corpus_lemmata_N", "*.txt")
outfolder = join(wdir, "topicmodel", "corpus_segs", "")
target = 1000
sizetolerancefactor = 1.1 # 1 = exact target; >1 = with some tolerance (1.1 = +/- 10%).
preserveparagraphs = False # True|False
prepare.segmenter(inpath, outfolder, target, sizetolerancefactor, preserveparagraphs)

### segments_to_bins: inpath, outfile
inpath = join(wdir, "topicmodel", "corpus_segs", "*.txt")
outfolder = join(wdir, "topicmodel", "3_bins") 
binsnb = 5  
#prepare.segments_to_bins(inpath, outfolder, binsnb)


### call_treetagger
### Perform lemmatization and POS tagging.
infolder = join(wdir, "txt_full")
outfolder = join(wdir, "topicmodel/corpus_tagged")
tagger = "/home/ulrike/Programme/tree-tagger-linux-3.2.1/cmd/tree-tagger-spanish"
#prepare.call_treetagger(infolder, outfolder, tagger) 

### make_lemmatext
### Extract selected lemmata from tagged text.
inpath = join(wdir, "topicmodel/corpus_tagged", "*.trt")
outfolder = join(wdir, "topicmodel/corpus_lemmata_N")
mode = "esN"
stoplist_errors = join(wdir, "features/topics_stopwords.txt") # in tmw folder
#prepare.make_lemmatext(inpath, outfolder, mode, stoplist_errors)


