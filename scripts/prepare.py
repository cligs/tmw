	#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: prepare.py
# Authors: christofs, daschloer, hennyu
# Version 0.3.0 (2016-03-20)

##################################################################
###  Topic Modeling Workflow (tmw):                            ###
##################################################################

##################################################################
###  prepare.py: preprocessing text files                      ###
##################################################################

import re
import os
import glob
import pandas as pd
from os import listdir
from os.path import join
from nltk.tokenize import word_tokenize
import glob
import subprocess
#from lxml import etree

    

#################################
# Segmenter                     #
#################################

# Utility function for writing segments
def writesegment(segment, outfolder, filename, counter, mode="w"):
    segname = join(outfolder, filename + "§{:04d}".format(counter) + ".txt")
    with open(segname, mode, encoding="utf-8") as output:
        output.write(' '.join(segment))
    output.close()



# Utility function for writing into files
def write(segment, file, mode = "w"):
    with open(file, mode, encoding="utf-8") as output:
        output.write(' '.join(segment))
        output.close()

# global segment counter
counter = 0
# global current segment size
currentsegmentsize = 0

# Utility function for writing segments
def writesegment(segment, outfolder, filename, target, tolerancefactor, preserveparagraphs):
    from os.path import join
    global currentsegmentsize
    global counter

    # ignore empty segments
    if segment == ["\n"] or len(segment) < 1:
        return

    # workaround for easy inter line-spacing in case of paragraph removal for lines combined into one segment
    if not preserveparagraphs and segment[-1] == "\n":
        segment = segment[0:len(segment) - 1]
        segment[-1] += " "
    segname = join(outfolder, filename + "§{:04d}".format(counter) + ".txt")
    relname = filename + "§{:04d}".format(counter) + ".txt"

    # case: last segment is too small => fill with (slice of) new segment
    if currentsegmentsize * tolerancefactor < target: # min size limit not reached => split
        #split segment
        wordsliceindex = target - currentsegmentsize

        # if it's too big: slice!
        if currentsegmentsize + len(segment) > target * tolerancefactor:
            #print(relname + "\t Last segment size: " + str(currentsegmentsize) + "\t appending " + str(wordsliceindex) + "\t for a total of " + str((currentsegmentsize + wordsliceindex)))
            write(segment[0:wordsliceindex], segname, "a")
            currentsegmentsize += wordsliceindex
            segment = segment[wordsliceindex:len(segment)]

            # segment is filled. continue with next one
            counter += 1
            currentsegmentsize = 0
            segname = join(outfolder, filename + "§{:04d}".format(counter) + ".txt")
            relname = filename + "§{:04d}".format(counter) + ".txt"
            if os.path.isfile(segname):
                os.remove(segname)
        # else just add text to current segment
        else:
            #print(relname + "\t Last segment size: " + str(currentsegmentsize) + "\t appending " + str(len(segment)) + "\t for a total of " + str((currentsegmentsize + len(segment))))
            # segment fits so append
            write(segment, segname, "a")
            currentsegmentsize += len(segment) - segment.count("\n") # take possible segment end into account!
            # done
            return

    # case: new segment is too big
    # if segment > target: slice segment
    while len(segment) > target * tolerancefactor:
        #print(relname + "\t Last segment size: " + str(currentsegmentsize) + "\t appending " + str(target) + "\t for a total of " + str((currentsegmentsize + target)))
        write(segment[0:target], segname)
        segment = segment[target:len(segment)]

        # segment is filled. continue with next one
        counter += 1
        currentsegmentsize = 0
        segname = join(outfolder, filename + "§{:04d}".format(counter) + ".txt")
        relname = filename + "§{:04d}".format(counter) + ".txt"
        if os.path.isfile(segname):
            os.remove(segname)
        #print(relname + "\t New segment with size \t0")
    # now size of segment is < target
    if (len(segment) == 0):
        #segment was perfectly sliced so we are done
        return

    # there's some part of segment left, write this into file

    # if the remaining part is exceeding current segment's capacity start new segment
    if currentsegmentsize + len(segment) > target * tolerancefactor:
        # segment is filled. continue with next one
        counter += 1
        currentsegmentsize = 0
        segname = join(outfolder, filename + "§{:04d}".format(counter) + ".txt")
        relname = filename + "§{:04d}".format(counter) + ".txt"
        if os.path.isfile(segname):
            os.remove(segname)
        #print(relname + "\t New segment with size \t0")
    #print(relname + "\t Last segment size: " + str(currentsegmentsize) + "\t appending " + str(len(segment)) + "\t for a total of " + str((currentsegmentsize + len(segment))))
    currentsegmentsize += len(segment) - segment.count("\n") # take possible segment end into account!
    write(segment, segname, "a")

def segmenter(inpath, outfolder, target, sizetolerancefactor, preserveparagraphs = False):
    """Script for turning plain text files into equal-sized segments, with limited respect for paragraph boundaries."""
    print("\nLaunched segmenter.")

    from os.path import join
    from nltk.tokenize import word_tokenize

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    global counter
    global currentsegmentsize
    # work on files in inpath
    for relfile in glob.glob(inpath):
        # get absolut filename
        file = join(inpath, relfile)
        with open(file, "r") as infile:
            filename = os.path.basename(file)[:-4]
            counter = 0
            currentsegmentsize = 0
            segname = join(outfolder, filename + "§{:06d}".format(counter) + ".txt")
            relname = filename + "§{:06d}".format(counter) + ".txt"
            if os.path.isfile(segname):
                os.remove(segname)
            # segment contains words assigned to the current segment
            segment = []

            # go through paragraphs one by one
            for line in infile:
                text = line
                # (optional) remove punctuation, special characters and space-chains
                #text = re.sub("[,;\.:!?¿\(\)—-]", " ", text)
                text = re.sub("[\t\r\n\v\f]", " ", text)
                text = re.sub("[ ]{1,9}", " ", text)

                # tokenize text
                words = word_tokenize(text)
                words.append("\n")
                writesegment(words, outfolder, filename, target, sizetolerancefactor, preserveparagraphs)
    print("Done.")


#################################
# call_treetagger               #
#################################

def call_treetagger(infolder, outfolder, tagger):
    """Function to call TreeTagger from Python"""
    print("\nLaunched call_treetagger.")
    inpath = join(infolder, "*.txt")
    infiles = glob.glob(inpath)
    counter = 0
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    for infile in infiles: 
        #print(os.path.basename(infile))
        counter+=1
        outfile = join(outfolder, os.path.basename(infile)[:-4] + ".trt")
        #print(outfile)
        command = tagger + " < " + infile + " > " + outfile
        subprocess.call(command, shell=True)
    print("Files treated: ", counter)
    print("Done.")
    
#################################
# make_lemmatext                #
#################################

def make_lemmatext(inpath, outfolder, mode, stoplist_errors):
    """Function to extract lemmas from TreeTagger output."""
    print("\nLaunched make_lemmatext.")

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    with open(stoplist_errors, "r", encoding="utf-8") as infile: 
        stoplist = infile.read()
    counter = 0
    for file in glob.glob(inpath):
        #print(os.path.basename(file))
        with open(file,"r", encoding="utf-8") as infile:
            counter+=1
            text = infile.read()
            splittext = re.split("\n",text)
            
            lemmata = []
            for line in splittext:
                splitline = re.split("\t",line)
                if len(splitline) == 3:
                    lemma = splitline[2]
                    pos = splitline[1]
                    token = splitline[0]
                    ## Select subset of lemmas according to parameter "mode"
                    if mode == "deN":
                        if "NN" in pos:
                            if "|" in lemma:
                                lemmata.append(token.lower())
                            elif "<unknown>" not in lemma:
                                lemmata.append(lemma.lower())
                    if mode == "deNV":
                        if "NN" in pos or "VV" in pos:
                            if "|" in lemma:
                                lemmata.append(token.lower())
                            elif "<unknown>" not in lemma:
                                lemmata.append(lemma.lower())
                    if mode == "deNVAdj":
                        if "NN" in pos or "VV" in pos or "ADJ" in pos:
                            if "|" in lemma:
                                lemmata.append(token.lower())
                            elif "<unknown>" not in lemma:
                                lemmata.append(lemma.lower())
                    if mode == "enN":
                        if "NN" in pos:
                            if "|" in lemma:
                                lemmata.append(token.lower())
                            elif "<unknown>" not in lemma:
                                lemmata.append(lemma.lower())
                    if mode == "enNV":
                        if "NN" in pos or "VB" in pos:
                            if "|" in lemma:
                                lemmata.append(token.lower())
                            elif "<unknown>" not in lemma:
                                lemmata.append(lemma.lower())
                    if mode == "frNVAA":
                        if "|" in lemma:
                            lemmata.append(token.lower())
                        elif "Nc" in pos or "Vv" in pos or "Rg" in pos or "Ag" in pos or "G" in pos:
                            if "<unknown>" not in lemma:
                                lemmata.append(lemma.lower())
                    if mode == "frNVA":
                        if "|" in lemma:
                            lemmata.append(token.lower())
                        elif "Nc" in pos or "Vv" in pos or "G" in pos or "Ag" in pos:
                            if "<unknown>" not in lemma:
                                lemmata.append(lemma.lower())
                    if mode == "ptN":
                        if "|" in lemma:
                            lemmata.append(token.lower())
                        elif "NC" in pos:
                            if "<unknown>" not in lemma:
                                lemmata.append(lemma.lower())
                            #else:
                            #    lemmata.append(token.lower())
                    if mode == "ptNV":
                        if "|" in lemma:
                            lemmata.append(token.lower())
                        elif "NC" in pos or "VM" in pos or "VA" in pos or "VS" in pos:
                            if "<unknown>" not in lemma:
                                lemmata.append(lemma.lower())
                    if mode == "ptNVAA":
                        if "|" in lemma:
                            lemmata.append(token.lower())
                        elif "NC" in pos or "VM" in pos or "VA" in pos or "VS" in pos or "RG" in pos or "RN" in pos or "AQ" in pos or "AO" in pos or "A0" in pos:
                            if "<unknown>" not in lemma:
                                lemmata.append(lemma.lower())
                    if mode == "itN":
                        if "|" in lemma:
                            lemmata.append(token.lower())
                        elif "NOM" in pos:
                            if "<unknown>" not in lemma:
                                lemmata.append(lemma.lower())
                    if mode == "itNV":
                        if "|" in lemma:
                            lemmata.append(token.lower())
                        elif "NOM" in pos or "VER" in pos:
                            if "<unknown>" not in lemma:
                                lemmata.append(lemma.lower())
                    if mode == "esN":   
                        if "|" in lemma:
                            lemmata.append(token.lower())
                        elif "NC" in pos or "NMEA" in pos:
                            if "<unknown>" not in lemma:
                                lemmata.append(lemma.lower())
                    if mode == "esNAAPP":
                        if "|" in lemma:
                            lemmata.append(token.lower())
                        elif "ADJ" in pos or "ADV" in pos or "NC" in pos or "NMEA" in pos or "PPC" in pos or "PPO" in pos or "PPX" in pos or "PREP" in pos or "PREP/DEL" in pos or "REL" in pos:
                            if "<unknown>" not in lemma:
                                lemmata.append(lemma.lower())
                    if mode == "esNVAAPP":
                        if "|" in lemma:
                            lemmata.append(token.lower())
                        elif "ADJ" in pos or "ADV" in pos or "NC" in pos or "NMEA" in pos or "PPC" in pos or "PPO" in pos or "PPX" in pos or "PREP" in pos or "PREP/DEL" in pos or "REL" in pos or "VM" in pos or "VA" in pos or "VS" in pos:
                            if "<unknown>" not in lemma:
                                lemmata.append(lemma.lower())
            ## Continue with list of lemmata, but remove undesired leftover words         
            lemmata = ' '.join([word for word in lemmata if word not in stoplist])
            lemmata = re.sub("[ ]{1,4}"," ", lemmata)
            lemmata = re.sub("","oe", lemmata)
            lemmata = re.sub("traire","trahir", lemmata)
            lemmata = re.sub("faut","falloir", lemmata)
            lemmata = re.sub("vois","voir", lemmata)
            #lemmata = re.sub("dorer","doré/dorer", lemmata)
            newfilename = os.path.basename(file)[:-4] + ".txt"
            #print(outfolder, newfilename)
            with open(join(outfolder, newfilename),"w", encoding="utf-8") as output:
                output.write(str(lemmata))
    print("Files treated: ", counter)
    print("Done.")



#################################
# create_stopword_list          #
#################################

def create_stopword_list(mfw, corpus_dir, stopwords_out):
    """
    Creates a stop word list for a collection of text files.
    The most frequent words of the collection are used as stop words.
    How many of the MFW should be used, can be indicated with the mfw parameter.

    author: uhk

    Arguments:
    mfw (int): number of MFW to consider as stop words
    corpus_dir (str): path to the corpus directory
    stopwords_out (str): path to the output stopwords file 
    
    """
    
    print("\nLaunched create_stopword_list.")
    
    from nltk.corpus import PlaintextCorpusReader
    from nltk.probability import FreqDist
    
    corpus = PlaintextCorpusReader(corpus_dir, '.*')
    fdist_corpus = FreqDist(corpus.words())
    
    with open(stopwords_out, "w", encoding="utf-8") as stopwords_out_file:
        
        # get the most frequent words
        mfw_list = [w[0] for w in fdist_corpus.most_common(mfw)]
        
        # write stop word list to file
        stopwords_out_file.write("\n".join(mfw_list))
        
    print("Done.")
	
	
