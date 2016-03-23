#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: prepare.py
# Authors: christofs, daschloer
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
from lxml import etree

#################################
# read TEI P5                   #
#################################

def read_tei5(teiPath, txtFolder, xpath):
    """
    Extracts selected text from TEI P5 files and writes TXT files.
    xpath (string): "alltext", "bodytext, "seg" or "said".
    """
    if not os.path.exists(txtFolder):
        os.makedirs(txtFolder)
    ## Do the following for each file in the inpath.
    counter = 0
    for file in glob.glob(teiPath):
        with open(file, "r"):
            filename = os.path.basename(file)[:-4]
            idno = filename[:6] # assumes idno is at the start of filename.
            #print("Treating " + idno)
            counter +=1
            xml = etree.parse(file)
            namespaces = {'tei':'http://www.tei-c.org/ns/1.0'}

            ### Removes tags but conserves their text content.
            ### USER: Uncomment as needed.
            etree.strip_tags(xml, "{http://www.tei-c.org/ns/1.0}seg")
            #etree.strip_tags(xml, "{http://www.tei-c.org/ns/1.0}said")
            #etree.strip_tags(xml, "{http://www.tei-c.org/ns/1.0}hi")

            ### Removes elements and their text content.
            ### USER: Uncomment as needed.
            #etree.strip_elements(xml, "{http://www.tei-c.org/ns/1.0}reg", with_tail=False)
            #etree.strip_elements(xml, "{http://www.tei-c.org/ns/1.0}orig", with_tail=False)
            etree.strip_elements(xml, "{http://www.tei-c.org/ns/1.0}note", with_tail=False)
            etree.strip_elements(xml, "{http://www.tei-c.org/ns/1.0}quote", with_tail=False)
            #etree.strip_elements(xml, "{http://www.tei-c.org/ns/1.0}l", with_tail=False)
            #etree.strip_elements(xml, "{http://www.tei-c.org/ns/1.0}p", with_tail=False)
            etree.strip_elements(xml, "{http://www.tei-c.org/ns/1.0}head", with_tail=False)
            #etree.strip_elements(xml, "{http://www.tei-c.org/ns/1.0}stage", with_tail=False)
            etree.strip_elements(xml, "{http://www.tei-c.org/ns/1.0}speaker", with_tail=False)

            ### XPath defining which text to select
            xp_bodytext = "//tei:body//text()"
            xp_alltext = "//text()"
            xp_seg = "//tei:body//tei:seg//text()"
            xp_said = "//tei:body//tei:said//text()"
            
            ### Applying one of the above XPaths, based on parameter passed.
            ### USER: use on of the xpath values used here in the parameters.
            if xpath == "bodytext": 
                text = xml.xpath(xp_bodytext, namespaces=namespaces)
            if xpath == "alltext": 
                text = xml.xpath(xp_alltext, namespaces=namespaces)
            if xpath == "seg": 
                text = xml.xpath(xp_seg, namespaces=namespaces)
            if xpath == "said": 
                text = xml.xpath(xp_said, namespaces=namespaces)
            text = "\n".join(text)

            ### Some cleaning up
            text = re.sub("[ ]{2,8}", " ", text)
            text = re.sub("\n{2,8}", "\n", text)
            text = re.sub("[ \n]{2,8}", " \n", text)
            text = re.sub("\t{1,8}", "\t", text)

            outtext = str(text)
            outfile = txtFolder + filename + ".txt"
        with open(outfile,"w") as output:
            output.write(outtext)
            
    print("Done. Files treated: " + str(counter))

    

#################################
# Segmenter                     #
#################################

# Utility function for writing segments
def writesegment(segment, outfolder, filename, counter, mode="w"):
    segname = join(outfolder, filename + "§{:04d}".format(counter) + ".txt")
    with open(segname, mode) as output:
        output.write(' '.join(segment))
    output.close()



# Utility function for writing into files
def write(segment, file, mode = "w"):
    with open(file, mode) as output:
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
# Binning                       #
#################################

def segments_to_bins(inpath, outfolder, binsnb):
    """
    Script for sorting text segments into bins.
    """
    print("\nLaunched segments_to_bins.")

    import math, sys
    import os
    import glob
    from collections import Counter

    ### Define various objects for later use.
    txtids = []
    segids = []

    filenames = []
    binids = []

    offset = sys.maxsize # used to track wrong segmenting (i.e. with segment numbering not starting with 0)

    ### Get filenames, text identifiers, segment identifiers.
    for file in glob.glob(inpath):
        filename = os.path.basename(file)[:-4]
        txtid = filename[:6]
        txtids.append(txtid)
        segid = filename[-4:]
        #print(filename, txtid, segid)
        segids.append(segid)
        offset = min(offset, int(segid))
    #txtids_sr = pd.Series(txtids)
    #segids_sr = pd.Series(segids)

    if offset > 0:
        print("Warning! Segment numbering should start at 0. Using offset: " + str(offset))

    ### For each text identifier, get number of segments.
    txtids_ct = Counter(txtids)
    sum_segnbs = 0
    for txtid in txtids_ct:
        segnb = txtids_ct[txtid]
        #print(segnb)
        sum_segnbs = sum_segnbs + segnb
        #print(txtid, segnb)
    print("Total number of segments: ", sum_segnbs)

    for txtid in txtids_ct:
        countsegs = txtids_ct[txtid]
        if binsnb > int(countsegs):
            print("Warning! You are expecting more bins than segments available! Bins will not be filled continuously!")

    ### Match each filename to the number of segments of the text.

    bcount = dict()
    for i in range(0, binsnb):
        bcount[i] = 0

    for file in glob.glob(inpath):
        filename = os.path.basename(file)[:-4]
        for txtid in txtids_ct:
            if txtid in filename:
                filename = filename + "$" + str(txtids_ct[txtid])
                #print(filename)

    ### For each filename, compute and append bin number
        txtid = filename[0:6]
        segid = filename[7:11]
        segnb = filename[12:]
        #print(txtid,segid,segnb)
        binid = ""

        segprop = (int(segid) - offset) / int(segnb)
        #print(txtid, segid, segnb, segprop)


        binid = math.floor(segprop * binsnb)

        if binid == binsnb: # avoid 1.0 beeing in seperate bin (should never happen due to offset!)
            print("Error: Segment numbering is wrong! Continuing anyway...")
            binid -= 1

        bcount[binid] += 1

        #print(segprop, binid)

        filenames.append(filename[:11])
        binids.append(binid)
    filenames_sr = pd.Series(filenames, name="segmentID")
    binids_sr = pd.Series(binids, name="binID")
    files_and_bins = pd.concat([filenames_sr,binids_sr], axis=1)
    print("chunks per bin: ", bcount)

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    outfile = outfolder+"segs-and-bins.csv"
    with open(outfile, "w") as outfile:
        files_and_bins.to_csv(outfile, index=False)


#################################
# call_treetagger               #
#################################

def call_treetagger(infolder, outfolder, tagger):
    """Function to call TreeTagger from Python"""
    print("\nLaunched call_treetagger.")

    import os
    import glob

    inpath = infolder + "*.txt"
    infiles = glob.glob(inpath)
    counter = 0
    
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    for infile in infiles: 
        #print(os.path.basename(infile))
        counter+=1
        outfile = outfolder + os.path.basename(infile)[:-4] + ".trt"
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

    import re
    import os
    import glob

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    with open(stoplist_errors, "r") as infile: 
        stoplist = infile.read()
    counter = 0
    for file in glob.glob(inpath):
        #print(os.path.basename(file))
        with open(file,"r") as infile:
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
                    if mode == "frN":
                        if "|" in lemma:
                            lemmata.append(token.lower())
                        elif "NOM" in pos and "|" not in lemma and "<unknown>" not in lemma:
                            lemmata.append(lemma.lower())
                    elif mode == "frNV":
                        if "|" in lemma:
                            lemmata.append(token.lower())
                        elif "NOM" in pos or "VER" in pos and "|" not in lemma and "<unknown>" not in lemma:
                            lemmata.append(lemma.lower())
                    elif mode == "frNVAA":
                        if "|" in lemma:
                            lemmata.append(token.lower())
                        elif "NOM" in pos or "VER" in pos or "ADJ" in pos or "ADV" in pos and "|" not in lemma and "<unknown>" not in lemma:
                            lemmata.append(lemma.lower())
                    elif mode == "esN":
                        if "|" in lemma and "NC" in pos:
                            lemmata.append(token.lower())
                        elif "NC" in pos and "|" not in lemma and "<unknown>" not in lemma:
                            lemmata.append(lemma.lower())
                    elif mode == "enNV":
                        if "NN" in pos or "VB" in pos and "|" not in lemma and "<unknown>" not in lemma:
                            lemmata.append(lemma.lower())
                    elif mode == "enN":
                        if "NN" in pos and "|" not in lemma and "<unknown>" not in lemma:
                            lemmata.append(lemma.lower())
            ## Continue with list of lemmata, but remove undesired leftover words         
            lemmata = ' '.join([word for word in lemmata if word not in stoplist])
            lemmata = re.sub("[ ]{1,4}"," ", lemmata)
            newfilename = os.path.basename(file)[:-4] + ".txt"
            #print(outfolder, newfilename)
            with open(os.path.join(outfolder, newfilename),"w") as output:
                output.write(str(lemmata))
    print("Files treated: ", counter)
    print("Done.")