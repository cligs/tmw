#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: my_tmw.py
# Author: #cf
# Version 0.2.0 (2015-08-27)

import prepare
from os.path import join

### Set the general working directory.
wdir = "/home/ulrike/Dokumente/GS/Veranstaltungen/WS16-17_Praxisworkshop"


### segmenter
### Split entire texts into smaller segments.
inpath = join(wdir, "1_txt", "*.txt")
outfolder = join(wdir, "2_segs")
target = 3000
sizetolerancefactor = 1 # 1 = exact target; >1 = with some tolerance (1.1 = +/- 10%).
preserveparagraphs = False # True|False
#prepare.segmenter(inpath, outfolder, target, sizetolerancefactor, preserveparagraphs)

### segments_to_bins: inpath, outfile
inpath = join(wdir, "2_segs", "*.txt")
outfolder = join(wdir, "3_bins") 
binsnb = 5	
#prepare.segments_to_bins(inpath, outfolder, binsnb)


### call_treetagger
### Perform lemmatization and POS tagging.
infolder = join(wdir, "2_segs")
outfolder = join(wdir, "4_tagged")
tagger = "/home/ulrike/Programme/tree-tagger-linux-3.2.1/cmd/tree-tagger-spanish"
#prepare.call_treetagger(infolder, outfolder, tagger) 

### make_lemmatext
### Extract selected lemmata from tagged text.
inpath = join(wdir, "4_tagged", "*.trt")
outfolder = join(wdir, "5_lemmata")
mode = "esN" # frN=nouns, esN=nouns, ptN=nouns, frNV=nouns+verbs, ptNV=nouns+verbs, frNVAA=nouns+verbs+adj+adverbs, ptNVAA=nouns+verbs+adj+adverbs 
stoplist_errors = join(wdir, "extras", "stopwords_errors_es.txt") # in tmw folder
#prepare.make_lemmatext(inpath, outfolder, mode, stoplist_errors)


