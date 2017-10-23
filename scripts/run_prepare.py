#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: my_tmw.py
# Author: #cf
# Version 0.2.0 (2015-08-27)

import prepare
from os.path import join

### Set the general working directory.
wdir = "/home/ulrike/Dokumente/GS/Veranstaltungen/2017_Verona/exercises_English"


### segmenter
### Split entire texts into smaller segments.
inpath = join(wdir, "corpus_txt", "*.txt")
outfolder = join(wdir, "data", "2_segs", "")
target = 2000
sizetolerancefactor = 1 # 1 = exact target; >1 = with some tolerance (1.1 = +/- 10%).
preserveparagraphs = False # True|False
#prepare.segmenter(inpath, outfolder, target, sizetolerancefactor, preserveparagraphs)

### segments_to_bins: inpath, outfile
inpath = join(wdir, "data", "2_segs", "*.txt")
outfolder = join(wdir, "data", "3_bins") 
binsnb = 5  
#prepare.segments_to_bins(inpath, outfolder, binsnb)


### call_treetagger
### Perform lemmatization and POS tagging.
infolder = join(wdir, "corpus_txt")
outfolder = join(wdir, "4_tagged")
tagger = "/home/ulrike/Programme/tree-tagger-linux-3.2.1/cmd/tree-tagger-english"
#prepare.call_treetagger(infolder, outfolder, tagger) 

### make_lemmatext
### Extract selected lemmata from tagged text.
inpath = join(wdir, "4_tagged", "*.trt")
outfolder = join(wdir, "5_lemmata_N")
mode = "enN"
stoplist_errors = join(wdir, "en_stopwords.txt") # in tmw folder
prepare.make_lemmatext(inpath, outfolder, mode, stoplist_errors)


