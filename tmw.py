#!/usr/bin/env python3
# Filename: tmw.py
# Author: #cf

##################################################################
###  Topic Modeling Workflow (tmw)                             ###
##################################################################

# TODO: Use os.path.join everywhere for cross-platform compatibility.


import re
import os
import glob
import pandas as pd


##################################################################
###  PREPROCESSING                                             ###
##################################################################


#################################
# tei5reader                    #
#################################

def tei5reader_fulldocs(inpath, outfolder):
    """Script for reading selected text from TEI P5 files."""
    print("\nLaunched tei5reader_fulldocs.")

    from lxml import etree
    #print("Using LXML version: ", etree.LXML_VERSION)

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
     
    for file in glob.glob(inpath):
        with open(file, "r"):
            filename = os.path.basename(file)[:-4]
            #print(filename[:5]) # = idno

            ### The following options may help with parsing errors.
            #parser = etree.XMLParser(collect_ids=False, recover=True)
            parser = etree.XMLParser(recover=True)
            xml = etree.parse(file, parser)
            
            ### The TEI P5 files do have a default namespace.
            namespaces = {'tei':'http://www.tei-c.org/ns/1.0'}

            ### Removes tags but conserves their text content.
            etree.strip_tags(xml, "{http://www.tei-c.org/ns/1.0}seg")

            ### Removes elements and their text content.
            #etree.strip_elements(xml, "speaker")
            etree.strip_elements(xml, "{http://www.tei-c.org/ns/1.0}note")
            #etree.strip_elements(xml, "stage")
            etree.strip_elements(xml, "{http://www.tei-c.org/ns/1.0}head")

            ### XPath defining which text to select
            xp_bodytext = "//tei:body//text()"
            #xp_alltext = "//text()"

            ### Applying one of the above XPaths
            text = xml.xpath(xp_bodytext, namespaces=namespaces)
            text = "\n".join(text)

            ### Some cleaning up
            text = re.sub("[ ]{1,20}", " ", text)
            text = re.sub("\t\n", "\n", text)
            text = re.sub("\n{1,10}", "\n", text)
            text = re.sub("\n \n", "\n", text)
            text = re.sub("\n.\n", "\n", text)
            text = re.sub("[ ]{1,20}", " ", text)

            outtext = str(text)
            outfile = outfolder + filename + ".txt"
        with open(outfile,"w") as output:
            output.write(outtext)
    print("Done.")
    


#################################
# segmenter                     #
#################################

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
            # go thru paragraphs one by one
            for line in infile:
                text = line
                # remove special characters and space-chains
                text = re.sub("[,;\.!?—\t\r\n\v\f]", " ", text)
                text = re.sub("-", " ", text)
                text = re.sub("[ ]{1,9}", " ", text)
                # tokenize text
                words = word_tokenize(text)
                words.append("\n")
                writesegment(words, outfolder, filename, target, sizetolerancefactor, preserveparagraphs)
    print("Done.")



#################################
# Binning                       #
#################################

# TODO: Rewrite entirely to make compatible with mastermatrix.



#################################
# pretokenize                   #
#################################

import csv

def perform_multipleSubs(substitutionsFile, text):
    """Search and replace from a table of string pairs."""
    ## With code from http://stackoverflow.com/users/735204/emmett-j-butler
    ## Load table and turn into dict
    with open(substitutionsFile, "r") as subsFile: 
        subs = csv.reader(subsFile)
        for rows in subs:
            subsDict = {rows[0]:rows[1] for rows in subs}
        #print(subsDict)
        ## Create a regular expression  from the dictionary keys
        regex = re.compile("(%s)" % "|".join(map(re.escape, subsDict.keys())))
        ## For each match, look-up corresponding value in dictionary
        return regex.sub(lambda mo: subsDict[mo.string[mo.start():mo.end()]], text) 

def pretokenize(inpath, substitutionsFile, outfolder):
    """Deletion of unwanted elided and hyphenated words for better tokenization in TreeTagger. Optional."""
    print("\nLaunched pretokenize.")
    for file in glob.glob(inpath):
        with open(file,"r") as text:
            text = text.read()
            text = perform_multipleSubs(substitutionsFile, text)
            basename = os.path.basename(file)
            cleanfilename = basename
            if not os.path.exists(outfolder):
                os.makedirs(outfolder)
        with open(os.path.join(outfolder, cleanfilename),"w") as output:
            output.write(text)
    print("Done.")



#################################
# call_treetagger               #
#################################

def call_treetagger(infolder, outfolder, tagger):
    """Function to call TreeTagger from Python"""
    print("\nLaunched call_treetagger.")

    import os
    import glob
    import subprocess

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
                        if "|" in lemma:
                            lemmata.append(token.lower())
                        elif "NC" in pos and "|" not in lemma and "<unknown>" not in lemma:
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



##################################################################
### TOPIC MODELLING WITH MALLET                                ###
##################################################################

# TODO: Concatenate two stoplists first, one for errors, one for deliberate ommissions.


#################################
# call_mallet_import            #
#################################


def call_mallet_import(mallet_path, infolder,outfolder, outfile, stoplist_project):
    """Function to import text data into Mallet."""
    print("\nLaunched call_mallet_import.")    
    import subprocess
    import os
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)    
    ### Fixed parameters.
    token_regex = "'\p{L}[\p{L}\p{P}]*\p{L}'"
    ### Building the command line command    
    command = mallet_path + " import-dir --input " + infolder + " --output " + outfile + " --keep-sequence --token-regex " + token_regex + " --remove-stopwords TRUE --stoplist-file " + stoplist_project
    ## Make the call 
    subprocess.call(command, shell=True)
    print("Done.\n")



#################################
# call_mallet_modeling          #
#################################

def call_mallet_modeling(mallet_path, inputfile,outfolder,num_topics,optimize_interval,num_iterations,num_top_words,doc_topics_max):
    """Function to perform topic modeling with Mallet."""
    print("\nLaunched call_mallet_modeling.")

    import os
    import subprocess
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    ### Fixed parameters    
    word_topics_counts_file = outfolder + "words-by-topics.txt"
    topic_word_weights_file = outfolder + "word-weights.txt"
    output_topic_keys = outfolder + "topics-with-words.csv"
    output_doc_topics = outfolder + "topics-in-texts.csv"
    output_topic_state = outfolder + "topic_state.gz"
    
    ### Constructing Mallet command from parameters.
    command = mallet_path +" train-topics --input "+ inputfile +" --num-topics "+ num_topics +" --optimize-interval "+ optimize_interval +" --num-iterations " + num_iterations +" --num-top-words " + num_top_words +" --word-topic-counts-file "+ word_topics_counts_file + " --topic-word-weights-file "+ topic_word_weights_file +" --output-state topic-state.gz"+" --output-topic-keys "+ output_topic_keys +" --output-doc-topics "+ output_doc_topics +" --doc-topics-max "+ doc_topics_max + " --output-state " + output_topic_state
    #print(command)
    subprocess.call(command, shell=True)
    print("Done.\n")




##################################################################
### POSTPROCESSING OF RAW DATA                                 ###
##################################################################



##############################
# create_mastermatrix        #
##############################

import numpy as np
import pandas as pd
import os
import glob

def get_metadata(metadatafile):
    print("  Getting metadata...")
    """Read metadata file and create DataFrame."""
    metadata = pd.DataFrame.from_csv(metadatafile, header=0, sep=",")
    #print("metadata\n", metadata)
    return metadata

def get_topicscores(topics_in_texts, number_of_topics): 
    """Create a matrix of segments x topics, with topic score values, from Mallet output.""" 
    print("  Getting topicscores...")   
    ## Load Mallet output (strange format)
    topicsintexts = pd.read_csv(topics_in_texts, header=None, skiprows=[0], sep="\t", index_col=0)
    #topicsintexts = topicsintexts.iloc[0:100,]  ### For testing only!!
    #print("topicsintexts\n", topicsintexts.head())
    listofsegmentscores = []
    idnos = []
    i = -1
    ## For each row, collect segment and idno
    for row_index, row in topicsintexts.iterrows():
        segment = row[1][-15:-4]
        idno = row[1][-15:-11]
        #print(segment, idno)
        idnos.append(idno)
        topics = []
        scores = []
        ## For each segment, get the topic number and its score
        i +=1
        for j in range(1,number_of_topics,2):
            k = j+1
            topic = topicsintexts.iloc[i,j]
            score = topicsintexts.iloc[i,k]
            #score = round(score, 4) ## round off for smaller file.
            topics.append(topic)
            scores.append(score)
        ## Create dictionary of topics and scores for one segment
        persegment = dict(zip(topics, scores))
        segmentscores = pd.DataFrame.from_dict(persegment, orient="index")
        segmentscores.columns = [segment]
        segmentscores = segmentscores.T
        listofsegmentscores.append(segmentscores)
    ## Putting it all together
    topicscores = pd.concat(listofsegmentscores)
    topicscores["segmentID"] = topicscores.index
    topicscores.fillna(0,inplace=True)
    #print("topicscores\n", topicscores)
    return topicscores
        
def get_docmatrix(corpuspath):
    """Create a matrix containing segments with their idnos."""
    print("  Getting docmatrix...")
    ## Create dataframe with filenames of segments and corresponding idnos.
    segs = []
    idnos = []
    for file in glob.glob(corpuspath): 
        seg,ext = os.path.basename(file).split(".")
        segs.append(seg)
        idno = seg[0:6]
        idnos.append(idno)
    docmatrix = pd.DataFrame(segs)
    docmatrix["idno"] = idnos
    docmatrix.rename(columns={0:"segmentID"}, inplace=True)
    #print("docmatrix\n", docmatrix)
    return docmatrix
    
def merge_data(corpuspath, metadatafile, topics_in_texts, mastermatrixfile, 
               number_of_topics):
    """Merges the three dataframes into one mastermatrix."""
    print("  Getting data...")
    ## Get all necessary data.
    metadata = get_metadata(metadatafile)
    docmatrix = get_docmatrix(corpuspath)
    topicscores = get_topicscores(topics_in_texts, number_of_topics)
    ## For inspection only.
    #print("Metadata\n", metadata.head())
    #print("Docmatrix\n", docmatrix.head())
    #print("topicscores\n", topicscores.head())
    print("  Merging data...")    
    ## Merge metadata and docmatrix, matching each segment to its metadata.
    mastermatrix = pd.merge(docmatrix, metadata, how="inner", on="idno")  
    #print("mastermatrix: metadata and docmatrix\n", mastermatrix)
    ## Merge mastermatrix and topicscores, matching each segment to its topic scores.
    #print(mastermatrix.columns)
    #print(topicscores.columns)
    #print(topicscores)
    mastermatrix = pd.merge(mastermatrix, topicscores, on="segmentID", how="inner")
    #print("mastermatrix: all three\n", mastermatrix.head())
    return mastermatrix

def create_mastermatrix(corpuspath, outfolder, mastermatrixfile, metadatafile, 
                        topics_in_texts, number_of_topics):
    """Builds the mastermatrix uniting all information about texts and topic scores."""
    print("\nLaunched create_mastermatrix.")
    print("(Warning: This is very memory-intensive and may take a while.)")
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    mastermatrix = merge_data(corpuspath, metadatafile, topics_in_texts, 
                              mastermatrixfile, number_of_topics)
    mastermatrix.to_csv(outfolder+mastermatrixfile, sep=",", encoding="utf-8")
    print("  Saved mastermatrix. Segments and columns:", mastermatrix.shape)    



################################
# calculate_averageTopicScores #
################################

def calculate_averageTopicScores(mastermatrixfile, targets, outfolder):
    """Function to calculate average topic scores based on the mastermatrix."""
    print("\nLaunched calculate_averageTopicScores.")
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    with open(mastermatrixfile, "r") as infile:
        mastermatrix = pd.DataFrame.from_csv(infile, header=0, sep=",")
    ## Calculate average topic scores for each target category 
    for target in targets:
        grouped = mastermatrix.groupby(target, axis=0)
        avg_topicscores = grouped.agg(np.mean)
        avg_topicscores = avg_topicscores.drop(["year"], axis=1)
        #avg_topicscores = avg_topicscores.drop(["tei"], axis=1)
        ## Save grouped averages to CSV file for visualization.
        resultfilename = "avgtopicscores_by-"+target+".csv"
        resultfilepath = outfolder+resultfilename
        ## TODO: Some reformatting here, or adapt make_heatmaps.
        avg_topicscores.to_csv(resultfilepath, sep=",", encoding="utf-8")
        print("  Saved average topic scores for:", target)    
    print("Done.")



#################################
# save_firstWords               #
#################################

def save_firstWords(topicWordFile, outfolder, filename):
    """Save a table of topics with their three most important words for each topic."""
    print("Launched save_someFirstWords.")
    with open(topicWordFile, "r") as infile:
        firstWords = {}
        topicWords = pd.read_csv(infile, sep="\t", header=None)
        topicWords = topicWords.drop(1, axis=1)
        topicWords = topicWords.iloc[:,1:2]
        topics = topicWords.index.tolist()
        words = []
        for topic in topics:
            topic = int(topic)
            row = topicWords.loc[topic]
            row = row[2].split(" ")
            row = str(row[0]+"-"+row[1]+"-"+row[2]+" ("+str(topic)+")")
            words.append(row)
        firstWords = dict(zip(topics, words))
        firstWordsSeries = pd.Series(firstWords, name="firstWords")
        #firstWordsSeries.index.name = "topic"
        #firstWordsSeries = firstWordsSeries.rename(columns = {'two':'new_name'})
        firstWordsSeries.reindex_axis(["firstwords"])
        print(firstWordsSeries)
        ## Saving the file.
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
        outfile = outfolder + filename
        with open(outfile, "w") as outfile: 
            firstWordsSeries.to_csv(outfile)
        print("Done.")



##################################################################
###    VISUALIZATION                                           ###
##################################################################

import matplotlib.pyplot as plt


#################################
# make_wordle_from_mallet       #
#################################

from wordcloud import WordCloud
import random


def read_mallet_output(word_weights_file):
    """Reads Mallet output (topics with words and word weights) into dataframe.""" 
    word_scores = pd.read_table(word_weights_file, header=None, sep="\t")
    word_scores = word_scores.sort(columns=[0,2], axis=0, ascending=[True, False])
    word_scores_grouped = word_scores.groupby(0)
    #print(word_scores.head())
    return word_scores_grouped

def get_wordlewords(words, word_weights_file, topic):
    """Transform Mallet output for wordle generation."""
    topic_word_scores = read_mallet_output(word_weights_file).get_group(topic)
    top_topic_word_scores = topic_word_scores.iloc[0:words]
    topic_words = top_topic_word_scores.loc[:,1].tolist()
    word_scores = top_topic_word_scores.loc[:,2].tolist()
    wordlewords = ""
    j = 0
    for word in topic_words:
        word = word
        score = word_scores[j]
        j += 1
        wordlewords = wordlewords + ((word + " ") * score)
    return wordlewords
        
def get_color_scale(word, font_size, position, orientation, random_state=None):
    """ Create color scheme for wordle."""
    #return "hsl(0, 00%, %d%%)" % random.randint(80, 100) # Greys for black background.
    return "hsl(221, 65%%, %d%%)" % random.randint(30, 35) # Dark blue for white background

def make_wordle_from_mallet(word_weights_file, 
                            topics,words,outfolder, 
                            font_path, dpi):
    """Generate wordles from Mallet output, using the wordcloud module."""
    print("\nLaunched make_wordle_from_mallet.")
    for topic in range(0,topics):
        ## Gets the text for one topic.
        text = get_wordlewords(words, word_weights_file, topic)
        wordcloud = WordCloud(font_path=font_path, background_color="white", margin=5).generate(text)
        default_colors = wordcloud.to_array()
        figure_title = "topic "+ str(topic)        
        plt.imshow(wordcloud.recolor(color_func=get_color_scale, random_state=3))
        plt.imshow(default_colors)
        plt.imshow(wordcloud)
        plt.title(figure_title, fontsize=24)
        plt.axis("off")
        
        ## Saving the image file.
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
        figure_filename = "wordle_tp"+"{:03d}".format(topic) + ".png"
        plt.savefig(outfolder + figure_filename, dpi=dpi)
        plt.close()
    print("Done.")

    
def crop_images(inpath, outfolder, left, upper, right, lower):
    """ Function to crop wordle files."""
    print("Launched crop_images.")
    from PIL import Image
    import glob
    import os

    counter = 0
    for file in glob.glob(inpath): 
        original = Image.open(file)
        filename = os.path.basename(file)[:-4]+"x.png"
        box = (left, upper, right, lower)
        cropped = original.crop(box)
        cropped.save(outfolder + filename)
        counter +=1
    print("Done. Images cropped:" , counter)



#################################
# plot_topTopics                #
#################################

# TODO: Move this one one level up if several plotting functions use it.
def get_firstWords(firstWordsFile):
    """Function to load list of top topic words into dataframe."""
    #print("  Getting firstWords.")
    with open(firstWordsFile, "r") as infile: 
        firstWords = pd.read_csv(infile, header=None)
        firstWords.drop(0, axis=1, inplace=True)
        firstWords.rename(columns={1:"topicwords"}, inplace=True)
        #print(firstWords)
        return(firstWords)

def get_targetItems(average, targetCategory):
    """Get a list of items included in the target category."""
    print(" Getting targetItems for: "+targetCategory)
    with open(average, "r") as infile:
        averageTopicScores = pd.DataFrame.from_csv(infile, sep=",")
        #print(averageTopicScores.head())
        targetItems = list(averageTopicScores.index.values)
        #print(targetItems)
        return(targetItems)    
     
def get_dataToPlot(average, firstWordsFile, topTopicsShown, item):
    """From average topic score data, select data to be plotted."""
    #print("  Getting dataToPlot.")
    with open(average, "r") as infile:
        ## Read the average topic score data
        allData = pd.DataFrame.from_csv(infile, sep=",")
        allData = allData.T
        ## Add top topic words to table for display later
        firstWords = get_firstWords(firstWordsFile)
        allData["firstWords"] = firstWords.iloc[:,0].values
        ## Create subset of data based on target.
        dataToPlot = allData[[item,"firstWords"]]
        dataToPlot = dataToPlot.sort(columns=item, ascending=False)
        dataToPlot = dataToPlot[0:topTopicsShown]
        dataToPlot = dataToPlot.set_index("firstWords")
        #print(dataToPlot)         
        return dataToPlot

def create_barchart_topTopics(dataToPlot, targetCategory, item, 
                              fontscale, height, dpi, outfolder):
    """Function to make a topTopics barchart."""
    print("  Creating plot for: "+item)
    ## Doing the plotting.
    dataToPlot.plot(kind="bar", legend=None) 
    plt.setp(plt.xticks()[1], rotation=90, fontsize = 11)   
    plt.title("Top-Topics für: "+item, fontsize=15)
    plt.ylabel("Scores", fontsize=13)
    plt.xlabel("Topics", fontsize=13)
    if height != 0:
        plt.ylim((0.000,height))
   
    ## Saving the plot to disk.
    outfolder = outfolder+targetCategory+"/"
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    figure_filename = outfolder+"tT_"+item+".png"
    plt.savefig(figure_filename, dpi=dpi)
    plt.close()

def plot_topTopics(averageDatasets, firstWordsFile, numberOfTopics, 
                   targetCategories, topTopicsShown, fontscale, 
                   height, dpi, outfolder): 
    """For each item in a category, plot the top n topics as a barchart."""
    print("Launched plot_topTopics.")
    for average in glob.glob(averageDatasets):
        for targetCategory in targetCategories: 
            if targetCategory in average:
                targetItems = get_targetItems(average, targetCategory)
                for item in targetItems:
                    dataToPlot = get_dataToPlot(average, firstWordsFile, topTopicsShown, item)
                    create_barchart_topTopics(dataToPlot, targetCategory, item, fontscale, height, dpi, outfolder)
    print("Done.")



#################################
# plot_topItems                 #
#################################

def get_topItems_firstWords(firstWordsFile, topic):
    """Function to load list of top topic words into dataframe."""
    #print("  Getting firstWords.")
    with open(firstWordsFile, "r") as infile: 
        firstWords = pd.DataFrame.from_csv(infile, header=None)
        firstWords.columns = ["firstWords"]
        # Only the words for one topic are needed.
        firstWords = firstWords.iloc[topic]
        firstWords = firstWords[0]
        return(firstWords)

def get_topItems_dataToPlot(average, firstWordsFile, topItemsShown, topic):
    """From average topic score data, select data to be plotted."""
    #print("  Getting dataToPlot.")
    with open(average, "r") as infile:
        ## Read the average topic score data
        allData = pd.DataFrame.from_csv(infile, sep=",")
        allData = allData.T
        ## Create subset of data based on target.
        dataToPlot = allData.iloc[topic,:]
        dataToPlot = dataToPlot.order(ascending=False)
        dataToPlot = dataToPlot[0:topItemsShown]
        #print(dataToPlot)
        return dataToPlot

def create_topItems_barchart(dataToPlot, firstWords, targetCategory, topic, 
                              fontscale, height, dpi, outfolder):
    """Function to make a topItems barchart."""
    print("  Creating plot for topic: "+str(topic))
    ## Doing the plotting.
    dataToPlot.plot(kind="bar", legend=None) 
    plt.title("Top "+targetCategory+" für topic: "+str(firstWords), fontsize=15)
    plt.ylabel("Scores", fontsize=13)
    plt.xlabel(targetCategory, fontsize=13)
    plt.setp(plt.xticks()[1], rotation=90, fontsize = 11)   
    if height != 0:
        plt.ylim((0.000,height))
    plt.tight_layout() 

    ## Saving the plot to disk.
    outfolder = outfolder+targetCategory+"/"
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    figure_filename = outfolder+"tI_by-"+targetCategory+"-{:03d}".format(topic)+".png"
    plt.savefig(figure_filename, dpi=dpi)
    plt.close()


def plot_topItems(averageDatasets, 
                  outfolder, 
                  firstWordsFile,  
                  numberOfTopics, 
                  targetCategories, 
                  topItemsShown, 
                  fontscale, 
                  height, 
                  dpi): 
    """Visualize topic score distribution data as barchart. """
    print("Launched plot_topItems")
    for average in glob.glob(averageDatasets):
        for targetCategory in targetCategories:
            if targetCategory in average:
                print(" Plotting for: "+targetCategory)
                topics = list(range(0,numberOfTopics))
                for topic in topics:
                    firstWords = get_topItems_firstWords(firstWordsFile, 
                                                         topic)
                    dataToPlot = get_topItems_dataToPlot(average, 
                                                         firstWordsFile, 
                                                         topItemsShown, 
                                                         topic)
                    create_topItems_barchart(dataToPlot, 
                                             firstWords, 
                                             targetCategory, 
                                             topic, 
                                             fontscale, 
                                             height, 
                                             dpi, 
                                             outfolder)
    print("Done.")



#################################
# topic_distribution_heatmap    #
#################################

import seaborn as sns

# TODO: This next function could be merged with above.
def get_heatmap_firstWords(firstWordsFile):
    """Function to load list of top topic words into dataframe."""
    #print("  Getting firstWords.")
    with open(firstWordsFile, "r") as infile: 
        firstWords = pd.read_csv(infile, header=None)
        firstWords.drop(0, axis=1, inplace=True)
        firstWords.rename(columns={1:"topicwords"}, inplace=True)
        #print(firstWords)
        return(firstWords)

def get_heatmap_dataToPlot(average, firstWordsFile, topTopicsShown, 
                           numberOfTopics):
    """From average topic score data, select data to be plotted."""
    #print("  Getting dataToPlot.")
    with open(average, "r") as infile:
        ## Read the average topic score data
        allScores = pd.DataFrame.from_csv(infile, sep=",")
        allScores = allScores.T
        ## Create subset of data based on target.
        stdevs = allScores.std(axis=1)
        allScores = pd.concat([allScores, stdevs], axis=1)
        allScores = allScores.sort(columns=0, axis=0, ascending=False)
        someScores = allScores[0:topTopicsShown]
        someScores = someScores.drop(0, axis=1)
        ## Necessary step to align dtypes of indexes for concat.
        someScores.index = someScores.index.astype(np.int64)        
        #print("dtype firstWords: ", type(firstWords.index))
        #print("dtype someScores: ", type(someScores.index))
        #print("\n==intersection==\n",someScores.index.intersection(firstWords.index))
        ## Add top topic words to table for display later
        firstWords = get_heatmap_firstWords(firstWordsFile)
        dataToPlot = pd.concat([someScores, firstWords], axis=1, join="inner")
        dataToPlot = dataToPlot.set_index("topicwords")
        #print(dataToPlot)
        ## Optionally, limit display to part of the columns
        #dataToPlot = dataToPlot.iloc[:,0:40]
        #print(dataToPlot)
        return dataToPlot

def create_distinctiveness_heatmap(dataToPlot, 
                                   topTopicsShown,
                                   targetCategory, 
                                   fontscale,
                                   dpi, 
                                   outfolder):

    sns.set_context("poster", font_scale=fontscale)
    sns.heatmap(dataToPlot, annot=False, cmap="YlOrRd", square=False)
    # Nice: bone_r, copper_r, PuBu, OrRd, GnBu, BuGn, YlOrRd
    plt.title("Verteilung der Topic Scores", fontsize=20)
    plt.xlabel(targetCategory, fontsize=16)
    plt.ylabel("Top topics (stdev)", fontsize=16)
    plt.setp(plt.xticks()[1], rotation=90, fontsize = 12)   
    plt.tight_layout() 

    ## Saving the plot to disk.
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    figure_filename = outfolder+"dist-heatmap_by-"+str(targetCategory)+".png"
    plt.savefig(figure_filename, dpi=dpi)
    plt.close()



def plot_distinctiveness_heatmap(averageDatasets, 
                                 firstWordsFile, 
                                 outfolder, 
                                 targetCategories, 
                                 numberOfTopics, 
                                 topTopicsShown, 
                                 fontscale, 
                                 dpi):
    """Visualize topic score distribution data as heatmap. """
    print("Launched plot_distinctiveness_heatmap.")
    for average in glob.glob(averageDatasets):
        for targetCategory in targetCategories: 
            if targetCategory in average and targetCategory != "segmentID":
                print(" Plotting for: "+targetCategory)
                dataToPlot = get_heatmap_dataToPlot(average, 
                                                    firstWordsFile, 
                                                    topTopicsShown,
                                                    numberOfTopics)
                create_distinctiveness_heatmap(dataToPlot, 
                                               topTopicsShown,
                                               targetCategory, 
                                               fontscale,
                                               dpi, 
                                               outfolder)



#################################
# plot_topicsOverTime           #
#################################

def get_overTime_firstWords(firstWordsFile):
    """Function to load list of top topic words into dataframe."""
    #print("  Getting firstWords.")
    with open(firstWordsFile, "r") as infile: 
        firstWords = pd.read_csv(infile, header=None)
        firstWords.drop(0, axis=1, inplace=True)
        firstWords.rename(columns={1:"topicwords"}, inplace=True)
        firstWords.index = firstWords.index.astype(np.int64)        
        #print(firstWords)
        return(firstWords)

def get_overTime_dataToPlot(average, firstWordsFile, entriesShown, topics): 
    """Function to build a dataframe with all data necessary for plotting."""
    #print("  Getting data to plot.")
    with open(average, "r") as infile:
        allScores = pd.DataFrame.from_csv(infile, sep=",")
        allScores = allScores.T        
        #print(allScores.head())
        ## Select the data for selected topics
        someScores = allScores.loc[topics,:]
        someScores.index = someScores.index.astype(np.int64)        
        ## Add information about the firstWords of topics
        firstWords = get_overTime_firstWords(firstWordsFile)
        dataToPlot = pd.concat([someScores, firstWords], axis=1, join="inner")
        dataToPlot = dataToPlot.set_index("topicwords")
        dataToPlot = dataToPlot.T
        #print(dataToPlot)
        return dataToPlot

def create_overTime_lineplot(dataToPlot, outfolder, fontscale, topics, dpi, height):
    """This function does the actual plotting and saving to disk."""
    print("  Creating lineplot for selected topics.")
    ## Plot the selected data
    dataToPlot.plot(kind="line", lw=3, marker="o")
    plt.title("Entwicklung der Topic Scores", fontsize=20)
    plt.ylabel("Topic scores (absolut)", fontsize=16)
    plt.xlabel("Jahrzehnte", fontsize=16)
    plt.setp(plt.xticks()[1], rotation=0, fontsize = 14)   
    if height != 0:
        plt.ylim((0.000,height))

    ## Saving the plot to disk.
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    ## Format the topic information for display
    topicsLabel = "-".join(str(topic) for topic in topics)
    figure_filename = outfolder+"lineplot-"+topicsLabel+".png"
    plt.savefig(figure_filename, dpi=dpi)
    plt.close()

def create_overTime_areaplot(dataToPlot, outfolder, fontscale, topics, dpi):
    """This function does the actual plotting and saving to disk."""
    print("  Creating areaplot for selected topics.")
    ## Turn absolute data into percentages.
    dataToPlot = dataToPlot.apply(lambda c: c / c.sum() * 100, axis=1)
    ## Plot the selected data
    dataToPlot.plot(kind="area")
    plt.title("Entwicklung der Topic Scores", fontsize=20)
    plt.ylabel("Topic scores (anteilig zueinander)", fontsize=16)
    plt.xlabel("Jahrzehnte", fontsize=16)
    plt.ylim((0,100))
    plt.setp(plt.xticks()[1], rotation=0, fontsize = 14)   

    ## Saving the plot to disk.
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    ## Format the topic information for display
    topicsLabel = "-".join(str(topic) for topic in topics)
    figure_filename = outfolder+"areaplot-"+topicsLabel+".png"
    plt.savefig(figure_filename, dpi=dpi)
    plt.close()

def plot_topicsOverTime(averageDatasets, firstWordsFile, outfolder, 
                        numberOfTopics, fontscale, dpi, height,  
                        mode, topics):
    """Function to plot development of topics over time using lineplots or areaplots."""
    print("Launched plot_topicsOverTime.")
    if mode == "line": 
        for average in glob.glob(averageDatasets):
            if "decade" in average:
                entriesShown = numberOfTopics
                dataToPlot = get_overTime_dataToPlot(average, firstWordsFile, 
                                                     entriesShown, topics)
                create_overTime_lineplot(dataToPlot, outfolder, fontscale, 
                                         topics, dpi, height)
    elif mode == "area":
        for average in glob.glob(averageDatasets):
            if "decade" in average:
                entriesShown = numberOfTopics
                dataToPlot = get_overTime_dataToPlot(average, firstWordsFile, 
                                                     entriesShown, topics)
                create_overTime_areaplot(dataToPlot, outfolder, fontscale, 
                                         topics, dpi)
    print("Done.")



###########################
## topic_clustering     ###
###########################

import scipy.cluster as sc

def get_topWordScores(wordWeightsFile, WordsPerTopic):
    """Reads Mallet output (topics with words and word weights) into dataframe.""" 
    print("- getting topWordScores...")
    wordScores = pd.read_table(wordWeightsFile, header=None, sep="\t")
    wordScores = wordScores.sort(columns=[0,2], axis=0, ascending=[True, False])
    topWordScores = wordScores.groupby(0).head(WordsPerTopic)
    #print(topWordScores)
    return topWordScores

def build_scoreMatrix(topWordScores, topicsToUse):
    """Transform Mallet output for wordle generation."""
    print("- building score matrix...")
    topWordScores = topWordScores.groupby(0)
    listOfWordScores = []
    for topic,data in topWordScores:
        if topic in list(range(0,topicsToUse)):
            words = data.loc[:,1].tolist()
            scores = data.loc[:,2].tolist()
            wordScores = dict(zip(words, scores))
            wordScores = pd.Series(wordScores, name=topic)
            listOfWordScores.append(wordScores)
        scoreMatrix = pd.concat(listOfWordScores, axis=1)
        scoreMatrix = scoreMatrix.fillna(10)
    #print(scoreMatrix.head)
    scoreMatrix = scoreMatrix.T
    return scoreMatrix

def perform_clustering(scoreMatrix, method, metric, wordsPerTopic, outfolder): 
    print("- performing clustering...")
    distanceMatrix = sc.hierarchy.linkage(scoreMatrix, method=method, metric=metric)
    #print(distanceMatrix)
    plt.figure(figsize=(25,10))
    sc.hierarchy.dendrogram(distanceMatrix)
    plt.setp(plt.xticks()[1], rotation=90, fontsize = 6)   
    plt.title("Topic-Clustering Dendrogramm", fontsize=20)
    plt.ylabel("Distanz", fontsize=16)
    plt.xlabel("Parameter: "+method+" clustering - "+metric+" distance measure - "+str(wordsPerTopic)+" words", fontsize=16)
    plt.tight_layout() 

    ## Saving the image file.
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    figure_filename = "clustering_"+metric+"-"+method+"-"+str(wordsPerTopic)+"words"+".png"
    plt.savefig(outfolder + figure_filename, dpi=600)
    plt.close()
    

def topicClustering(wordWeightsFile, wordsPerTopic, outfolder, 
                    methods, metrics, topicsToUse):
    """Display dendrogram of topic similarity using clustering."""
    print("\nLaunched topicClustering.")
    ## Gets the necessary data: the word scores for each topic
    topWordScores = get_topWordScores(wordWeightsFile, wordsPerTopic)
    ## Turn the data into a dataframe for further processing
    scoreMatrix = build_scoreMatrix(topWordScores, topicsToUse)
    ## Do clustering on the dataframe
    for method in methods: 
        for metric in metrics: 
            perform_clustering(scoreMatrix, method, metric, wordsPerTopic, outfolder)
    print("Done.")



##################################################################
###    OTHER / OBSOLETE                                        ###
##################################################################



###########################
## show_segment         ###
###########################

import shutil

def show_segment(wdir,segmentID, outfolder): 
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    shutil.copyfile(wdir+"2_segs/"+segmentID+".txt",outfolder+segmentID+".txt")