#!/usr/bin/env python3
# Filename: tmw.py

##################################################################
###  Topic Modeling Workflow (tmw)                             ###
##################################################################



##################################################################
###  1. Reading and segmenting texts                           ###
##################################################################

def tei5reader_fulldocs(inpath, outfolder):
    """Script for reading selected text from TEI P5 files."""
    print("\nLaunched tei5reader_fulldocs.")

    import re
    import os
    import glob
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
            etree.strip_tags(xml, "{http://www.tei-c.org/ns/1.0}hi")

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
            text = re.sub("  ", "", text)
            #text = re.sub("    ", "", text)
          #  text = re.sub("\n{1,6}", " ", text)
            #text = re.sub("\n{1,6}", "\n", text)
          #  text = re.sub("\n \n", "\n", text)
           # text = re.sub("\t\n", "", text)

            outtext = str(text)
            outfile = outfolder + filename + ".txt"
        with open(outfile,"w") as output:
            output.write(outtext)
    print("Done.")

# Utility function for writing segments
def writesegment(segment, outfolder, filename, counter, mode="w"):
    from os.path import join
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
            print(relname + "\t Last segment size: " + str(currentsegmentsize) + "\t appending " + str(wordsliceindex) + "\t for a total of " + str((currentsegmentsize + wordsliceindex)))
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
            print(relname + "\t Last segment size: " + str(currentsegmentsize) + "\t appending " + str(len(segment)) + "\t for a total of " + str((currentsegmentsize + len(segment))))
            # segment fits so append
            write(segment, segname, "a")
            currentsegmentsize += len(segment) - segment.count("\n") # take possible segment end into account!
            # done
            return

    # case: new segment is too big
    # if segment > target: slice segment
    while len(segment) > target * tolerancefactor:
        print(relname + "\t Last segment size: " + str(currentsegmentsize) + "\t appending " + str(target) + "\t for a total of " + str((currentsegmentsize + target)))
        write(segment[0:target], segname)
        segment = segment[target:len(segment)]

        # segment is filled. continue with next one
        counter += 1
        currentsegmentsize = 0
        segname = join(outfolder, filename + "§{:04d}".format(counter) + ".txt")
        relname = filename + "§{:04d}".format(counter) + ".txt"
        if os.path.isfile(segname):
            os.remove(segname)
        print(relname + "\t New segment with size \t0")

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
        print(relname + "\t New segment with size \t0")

    print(relname + "\t Last segment size: " + str(currentsegmentsize) + "\t appending " + str(len(segment)) + "\t for a total of " + str((currentsegmentsize + len(segment))))
    currentsegmentsize += len(segment) - segment.count("\n") # take possible segment end into account!
    write(segment, segname, "a")



# Parameters:
#   - inpath:               path to search documents in
#   - outfolder:            path to save segments in
#   - target:               number of words per segment
#   - sizetolerancefactor:  factor of which exceedance of target is tolerated before slicing paragraphs
#                               1 for zero tolerance
#                              -1 for infinity tolerance
#   - preserveparagraphs:   if True, segments will contain linebreaks according to paragraphs
#
def segmenter(inpath, outfolder, target, sizetolerancefactor, preserveparagraphs):
    """Script for turning plain text files into equal-sized segments, without respecting paragraph boundaries."""
    print("\nLaunched segmenter.")
    import os
    import re
    from os import listdir
    from os.path import join
    from nltk.tokenize import word_tokenize
    import glob

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
            segname = join(outfolder, filename + "§{:04d}".format(counter) + ".txt")
            relname = filename + "§{:04d}".format(counter) + ".txt"
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

                # tokanize text
                words = word_tokenize(text)

                words.append("\n")
                writesegment(words, outfolder, filename, target, sizetolerancefactor, preserveparagraphs)

    print("Done.")

def segments_to_bins(inpath, outfile):
    """Script for sorting text segments into bins."""
    print("\nLaunched segments_to_bins.")

    import os
    import glob
    from collections import Counter
    import pandas as pd

    ### Define various objects for later use.
    txtids = []
    segids = []
    #binsnb = 5
    filenames = []
    binids = []


    ### Get filenames, text identifiers, segment identifiers.
    for file in glob.glob(inpath):
        filename = os.path.basename(file)[:-4]
        txtid = filename[:6]
        txtids.append(txtid)
        segid = filename[-4:]
        #print(filename, txtid, segid)
        segids.append(segid)
    #txtids_sr = pd.Series(txtids)
    #segids_sr = pd.Series(segids)

    ### For each text identifier, get number of segments.
    txtids_ct = Counter(txtids)
    sum_segnbs = 0
    for txtid in txtids_ct:
        segnb = txtids_ct[txtid]
        #print(segnb)
        sum_segnbs = sum_segnbs + segnb
        #print(txtid, segnb)
    print("Total number of segments: ", sum_segnbs)


    ### Match each filename to the number of segments of the text.

    bcount0 = 0
    bcount1 = 0
    bcount2 = 0
    bcount3 = 0
    bcount4 = 0

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

        segprop = int(segid) / int(segnb)
        #print(txtid, segid, segnb, segprop)
        if segprop > 0 and segprop <= 0.21:
            binid = 1
            bcount0 += 1
        if segprop > 0.21 and segprop <= 0.41:
            binid = 2
            bcount1 += 1
        if segprop > 0.41 and segprop <= 0.61:
            binid = 3
            bcount2 += 1
        if segprop > 0.61 and segprop <= 0.81:
            binid = 4
            bcount3 += 1
        if segprop > 0.81 and segprop <= 1:
            binid = 5
            bcount4 += 1
        #print(segprop, binid)

        filenames.append(filename[:10])
        binids.append(binid)
    filenames_sr = pd.Series(filenames, name="filenames")
    binids_sr = pd.Series(binids, name="binids")
    files_and_bins = pd.concat([filenames_sr,binids_sr], axis=1)

    print("chunks per bin: ", bcount0,bcount1,bcount2,bcount3,bcount4)
    with open(outfile, "w") as outfile:
        files_and_bins.to_csv(outfile, index=False)

    print("Done.")



##################################################################
###  2. Preprocessing text segments                            ###
##################################################################



def pretokenize(inputpath,outputfolder):
    """Deletion of unwanted elided and hyphenated words for better tokenization in TreeTagger. Optional."""
    print("\nLaunched pretokenize.")

    import re
    import os
    import glob

    numberoffiles = 0
    for file in glob.glob(inputpath):
        numberoffiles +=1
        with open(file,"r") as text:
            text = text.read()

### Idea for future implementation of replacements
#        replacements = {"J'":"Je", "S'":"Se", "’":"'", "":""}
#        for item in replacements:
#            text = re.sub(replacements.key(), replacements.value(), text)

            text = re.sub("’","'",text)
            text = re.sub("J'","Je ",text)
            text = re.sub("j'","je ",text)
            text = re.sub("S'","Se ",text)
            text = re.sub("s'","se ",text)
            text = re.sub("C'","Ce ",text)
            text = re.sub("c'","ce ",text)
            text = re.sub("N'","Ne ",text)
            text = re.sub("n'","ne ",text)
            text = re.sub("D'","De ",text)
            text = re.sub("d'","de ",text)
            text = re.sub("L'","Le ",text)
            text = re.sub("l'","la ",text)
            text = re.sub("T'","tu|te ",text)
            text = re.sub("t'","tu|te ",text)
            text = re.sub("-le"," le",text)
            text = re.sub("-moi"," moi",text)
            text = re.sub("m'","me ",text)
            text = re.sub("M'","Me ",text)
            text = re.sub("-je"," je",text)
            text = re.sub("-il"," il",text)
            text = re.sub("-on"," on",text)
            text = re.sub("-lui"," lui",text)
            text = re.sub("-elle"," elle",text)
            text = re.sub("-nous"," nous",text)
            text = re.sub("-vous"," vous",text)
            text = re.sub("-nous"," nous",text)
            text = re.sub("-ce"," ce",text)
            text = re.sub("-tu"," tu",text)
            text = re.sub("-toi"," toi",text)
            text = re.sub("jusqu'à'","jusque à",text)
            text = re.sub("aujourd'hui","aujourdhui",text)
            text = re.sub("-t","",text)
            text = re.sub("-y"," y",text)
            text = re.sub("-en"," en",text)
            text = re.sub("-ci"," ci",text)
            text = re.sub("-là"," là",text)
            #text = re.sub("là-bas","là bas",text)
            text = re.sub("Qu'","Que ",text)
            text = re.sub("qu'","que ",text)
            text = re.sub("-même"," même",text)

            basename = os.path.basename(file)
            cleanfilename = basename
            #print(cleanfilename)
            if not os.path.exists(outputfolder):
                os.makedirs(outputfolder)
        with open(os.path.join(outputfolder, cleanfilename),"w") as output:
            output.write(text)
    #print("Number of files treated: " + str(numberoffiles))
    print("Done.")



def nltk_stanfordpos(inpath, outfolder):
    """POS-Tagging French text with Stanford POS-Tagger via NLTK."""
    print("\nLaunched nltk_stanfordpos.")

    import os
    import glob
    from nltk.tag.stanford import POSTagger

    for file in glob.glob(inpath):
        st = POSTagger('/home/christof/Programs/stanfordpos/models/french.tagger', '/home/christof/Programs/stanfordpos/stanford-postagger.jar', encoding="utf8")
        with open(file, "r", encoding="utf-8") as infile:
            untagged = infile.read()
            tagged = st.tag(untagged.split())

            taggedstring = ""
            for item in tagged:
                item = "\t".join(item)
                taggedstring = taggedstring + str(item) + "\n"
            #print(taggedstring)

            basename = os.path.basename(file)
            cleanfilename = basename
            if not os.path.exists(outfolder):
                os.makedirs(outfolder)
            with open(os.path.join(outfolder, cleanfilename),"w") as output:
                output.write(taggedstring)
    print("Done.")



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



def make_lemmatext(inpath,outfolder):
    """Function to extract lemmas from TreeTagger output."""
    print("\nLaunched make_lemmatext.")

    import re
    import os
    import glob

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
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
                    word = splitline[0]
                    if "|" in lemma:
                        lemmata.append(word.lower())
                    elif "NOM" in pos and "|" not in lemma and "<unknown>" not in lemma:
                    #elif "NOM" in pos or "VER" in pos or "ADJ" in pos or "ADV" in pos and "|" not in lemma and "<unknown>" not in lemma:
                        lemmata.append(lemma.lower())
            stoplist = ["les","suis","est","un", "pas", "abord", "rien", "fait", "ton", "moi","être"]
            lemmata = ' '.join([word for word in lemmata if word not in stoplist])
            lemmata = re.sub("[ ]{1,4}"," ", lemmata)
            newfilename = os.path.basename(file)[:-4] + ".txt"
            #print(outfolder, newfilename)
            with open(os.path.join(outfolder, newfilename),"w") as output:
                output.write(str(lemmata))
    print("Files treated: ", counter)
    print("Done.")



##################################################################
###  3. Importing and modeling with Mallet                     ###
##################################################################



def call_mallet_import(mallet_path, infolder,outfolder, outfile, stoplist):
    """Function to import text data into Mallet."""
    print("\nLaunched call_mallet_import.")
    
    import subprocess
    import os

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    
    ### Fixed parameters.
    token_regex = "'\p{L}[\p{L}\p{P}]*\p{L}'"
    
    command = mallet_path + " import-dir --input " + infolder + " --output " + outfile + " --keep-sequence --token-regex " + token_regex + " --remove-stopwords TRUE --stoplist-file " + stoplist
    #print(command)
    subprocess.call(command, shell=True)
    print("Done.\n")



def call_mallet_modeling(mallet_path, inputfile,outfolder,num_topics,optimize_interval,num_iterations,num_top_words,doc_topics_max):
    """Function to perform topic modeling with Mallet."""
    print("\nLaunched call_mallet_modeling.")

    ### Getting ready.
    import os
    import subprocess
    
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    ### Fixed parameters    
    word_topics_counts_file = outfolder + "words-by-topics.txt"
    topic_word_weights_file = outfolder + "word-weights.txt"
    output_topic_keys = outfolder + "topics-with-words.csv"
    output_doc_topics = outfolder + "topics-in-texts.csv"
    
    ### Constructing Mallet command from parameters.
    command = mallet_path +" train-topics --input "+ inputfile +" --num-topics "+ num_topics +" --optimize-interval "+ optimize_interval +" --num-iterations " + num_iterations +" --num-top-words " + num_top_words +" --word-topic-counts-file "+ word_topics_counts_file + " --topic-word-weights-file "+ topic_word_weights_file +" --output-state topic-state.gz"+" --output-topic-keys "+ output_topic_keys +" --output-doc-topics "+ output_doc_topics +" --doc-topics-max "+ doc_topics_max
    #print(command)
    subprocess.call(command, shell=True)
    print("Done.\n")


##################################################################
###  5. Aggregate data and visualize results                   ###
##################################################################



def make_wordle_from_mallet(word_weights_file,topics,words,outfolder, font_path, dpi):
    """Generate wordles from Mallet output, using the wordcloud module."""
    print("\nLaunched make_wordle_from_mallet.")
    
    import os
    import pandas as pd
    import random
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    
    def read_mallet_output(word_weights_file):
        """Reads Mallet output (topics with words and word weights) into dataframe.""" 
        word_scores = pd.read_table(word_weights_file, header=None, sep="\t")
        word_scores = word_scores.sort(columns=[0,2], axis=0, ascending=[True, False])
        word_scores_grouped = word_scores.groupby(0)
        #print(word_scores.head())
        return word_scores_grouped

    def get_wordlewords(words,topic):
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

    ## Creates the wordle visualisation, using results from the above functions.
    for topic in range(0,topics):
        ## Defines filename and title for the wordle image.
        figure_filename = "wordle_tp"+"{:03d}".format(topic) + ".png"
        figure_title = "topic "+ "{:02d}".format(topic)        
        ## Gets the text for one topic.
        text = get_wordlewords(words,topic)
        #print(text)
        ## Generates, recolors and saves the wordcloud.
        #original# wordcloud = WordCloud(background_color="white", margin=5).generate(text)
        #font_path = "/home/christof/.fonts/AveriaSans-Regular.ttf"
        wordcloud = WordCloud(font_path=font_path, background_color="white", margin=5).generate(text)
        default_colors = wordcloud.to_array()
        plt.imshow(wordcloud.recolor(color_func=get_color_scale, random_state=3))
        plt.imshow(default_colors)
        plt.imshow(wordcloud)
        plt.title(figure_title, fontsize=24)
        plt.axis("off")
        plt.savefig(outfolder + figure_filename, dpi=dpi)
        plt.close()
    print("Done.")
    


"""
# Average topicscores based on metadata.
"""

import numpy as np
import pandas as pd
import os
import glob

def average_topicscores(corpuspath, mastermatrixfile, metadatafile, topics_in_texts, targets, mode, number_of_topics, outfolder):
    """Function to calculate average topic scores based on metadata."""
    print("\nLaunched average_topicscores.")

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    ## Get the matrix of all data, either by creating a new one or by loading an existing one.
    if mode == "create": 
        print("  Creating new mastermatrix from data. This could take a while.")
        mastermatrix = merge_data(corpuspath, metadatafile, topics_in_texts, mastermatrixfile, number_of_topics)
    elif mode == "load":
        print("  Loading existing mastermatrix.")
        with open(mastermatrixfile, "r") as infile:
            mastermatrix = pd.DataFrame.from_csv(infile, header=0, sep=",")

    print("  Performing calculations...")
    ## Group by author, get median publication year and stdev per author.        
    #grouped = mastermatrix.groupby(target, axis=0)
    #publicationstats = grouped["year"].agg([np.median,np.std])
    #print(publicationstats)

    ## Calculate average topic scores for each target category 
    for target in targets:
        grouped = mastermatrix.groupby(target, axis=0)
        avg_topicscores = grouped.agg(np.mean)
        avg_topicscores = avg_topicscores.drop(["year"], axis=1)
        #print(avg_topicscores.head())
  
        ## Save grouped averages to CSV file for visualization.
        resultfilename = "avgtopicscores_by-"+target+".csv"
        resultfilepath = outfolder+resultfilename
        ## TODO: Some reformatting here, or adapt make_heatmaps.
        avg_topicscores.to_csv(resultfilepath, sep=",", encoding="utf-8")
        print("  Saved average topic scores for:", target)    
    print("Done.")

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
        idno = row[1][-15:-9]
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
    
def merge_data(corpuspath, metadatafile, topics_in_texts, mastermatrixfile, number_of_topics):
    """Merges the three dataframes into one mastermatrix."""
    print("  Getting data...")

    ## Get all necessary data.
    metadata = get_metadata(metadatafile)
    docmatrix = get_docmatrix(corpuspath)
    topicscores = get_topicscores(topics_in_texts, number_of_topics)

    print("  Merging data...")    
    ## Merge metadata and docmatrix, matching each segment to its metadata.
    mastermatrix = pd.merge(docmatrix, metadata, how="inner", on="idno")  
    #print("mastermatrix: metadata and docmatrix\n", mastermatrix)
    ## Merge mastermatrix and topicscores, matching each segment to its topic scores.
    #print(mastermatrix.columns)
    #print(topicscores.columns)
    #print(topicscores)
    mastermatrix = pd.merge(mastermatrix, topicscores, on="segmentID", how="inner")
    #print("mastermatrix: all three\n", mastermatrix)
    mastermatrix.to_csv(mastermatrixfile, sep=",", encoding="utf-8")
    print("  Saved mastermatrix. Segments and columns:", mastermatrix.shape)    
    return mastermatrix





def aggregate_using_bins_and_metadata(corpuspath,outfolder,topics_in_texts,metadatafile,bindatafile,target):
    """Aggregate topic scores based on positional bins and metadata."""
    print("\nLaunched aggregate_using_bins_and_metadata.")

    import numpy as np
    import itertools
    import operator
    import os
    import pandas as pd

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    ## USER: Set path to where the individual chunks are located.
    CORPUS_PATH = os.path.join(corpuspath)
    filenames = sorted([os.path.join(CORPUS_PATH, fn) for fn in os.listdir(CORPUS_PATH)])
    print("Number of files to treat: ", len(filenames)) #ok
    #print("First three filenames: ", filenames[:3]) #ok

    def grouper(n, iterable, fillvalue=None):
        "Collect data into fixed-length chunks or blocks"
        args = [iter(iterable)] * n
        return itertools.zip_longest(*args, fillvalue=fillvalue)

    doctopic_triples = []
    mallet_docnames = []
    ### USER: Set path to results from Mallet.
    with open(topics_in_texts) as f:
        f.readline()
        for line in f:
            docnum, docname, *values = line.rstrip().split('\t')
            mallet_docnames.append(docname)
            for topic, share in grouper(2, values):
                triple = (docname, int(topic), float(share))
                doctopic_triples.append(triple)

    doctopic_triples = sorted(doctopic_triples, key=operator.itemgetter(0,1))
    mallet_docnames = sorted(mallet_docnames)
    num_docs = len(mallet_docnames)
    num_topics = len(doctopic_triples) // len(mallet_docnames)
    print("Number of documents: ", num_docs)
    print("Number of topics: ", num_topics)

    # This creates a 2D array where each row is one document and each column gives the topic score for one topic for all documents. Building this tends to take a while.
    doctopic = np.zeros((num_docs, num_topics))
    counter = 0
    for triple in doctopic_triples:
        docname, topic, share = triple
        row_num = mallet_docnames.index(docname)
        doctopic[row_num, topic] = share
        counter += 1
        if counter % 50000 == 0:
            print("Iterations done:", counter)
    print("Uff. Done creating doctopic triples")
    #print(doctopic[0:1])


    #### Define aggregation criterion #
    metadata = pd.DataFrame.from_csv(metadatafile, header=0, sep=",")
    bindata = pd.DataFrame.from_csv(bindatafile, header=0, sep=",")
    #print(bindata.head())
    label_names = []
    for item in filenames:
        basename = os.path.basename(item)
        filename, ext = os.path.splitext(basename)
        textidno = filename[0:6]
        metadata_target = target
        genre_label = metadata.loc[textidno,metadata_target]
        binidno = filename[0:12]
        bin_target = "binids"
        bin_label = bindata.loc[binidno,bin_target]
        #print("textidno, binidno, genre_label, bin_label: ", textidno, binidno, genre_label, bin_label)
        #print(filename[0:1], bin_label)
        label_name = str(genre_label) + "$" + str(bin_label)
        outputfilename = outfolder + "topics_by_BINS-and "+ target.upper() + "-lp.csv"
        label_names.append(label_name)
    label_names = np.asarray(label_names)
    num_groups_labels = len(set(label_names))
    print("Number of entries: ", len(label_names))
    #print("Some label names: ", label_names[10:21])
    print("Number of different labels: ", len(set(label_names)))
    print("Number of topics: ", num_topics)

    ### Group topic scores according to label
    # Create 2D numpy array filled with zeros, and with space for labels x topics.
    doctopic_grouped = np.zeros((num_groups_labels, num_topics))
    # Fill up the array with topic scores you generate; 
    for i, name in enumerate(sorted(set(label_names))):
        #print("i and name: ", i, name)
        doctopic_grouped[i, :] = np.mean(doctopic[label_names == name, :], axis=0)
    doctopic = doctopic_grouped
    #print("Length of doctopic: ", len(doctopic)) #ok
    #np.savetxt("doctopic.csv", doctopic, delimiter=",")

    rownames = sorted(set(label_names))
    colnames = ["tp" + "{:03d}".format(i) for i in range(doctopic.shape[1])]
    df = pd.DataFrame(doctopic, index=rownames, columns=colnames)
    df.to_csv(outputfilename, sep='\t', encoding='utf-8')

    print("Done.")

# TODO: not necessary to write bin id onto filename (in "scenes_to_bins"), since it can be (and is) looked up in the bindatafile.
# TODO: Actually, this is even a problem when switching between scene-based and segment-based aggregation. Solution needed. 



"""
# make_topic_distribution_plots
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def make_topic_distribution_plot(aggregates,outfolder,topicwordfile,rows_shown,font_scale,dpi, mode, topics):
    """Function to coordinate creation of topic distribution heatmap."""
    print("Launched make_topic_distribution_heatmap.")
    for aggregate in glob.glob(aggregates):
        ## Get topic score distribution for each aggregation file.
        topicscores = get_topicscoredistribution(aggregate, rows_shown)
        ## For each topic appearing in topicscores, get the first three words.
        allfirstwords = get_firstwords(topicwordfile, topicscores)
        ## Set the first three words as the index of the topicscores
        topicscores["firstwords"] = allfirstwords
        #topicscores = topicscores.set_index("firstwords")
        ## Use acquired data to do actual visualisation.
        if mode == "heatmap": 
            create_heatmap(aggregate,topicscores,outfolder,rows_shown,font_scale,dpi)
        elif mode == "lineplot": 
            kind = "line"
            create_lineplot(aggregate,topicscores,outfolder,rows_shown,font_scale,dpi,topics,kind)
        elif mode == "areaplot":
            kind = "area"
            create_lineplot(aggregate,topicscores,outfolder,rows_shown,font_scale,dpi,topics,kind)
    print("Done.")

def get_topicscoredistribution(aggregate, rows_shown):
    with open(aggregate, "r") as infile:
        topicscores = pd.DataFrame.from_csv(infile, sep=",")
        topicscores = topicscores.T
        stdevs = topicscores.std(axis=1)
        topicscores = pd.concat([topicscores, stdevs], axis=1)
        topicscores = topicscores.sort(columns=0, axis=0, ascending=False)
        # column: 0=stdev; "seg1" = beginning, "Comédie", etc.
        topicscores = topicscores.iloc[:rows_shown,:-1] #rows,columns
        return topicscores

def get_firstwords(topicwordfile, topicscores):
    """Get three (or n) most important words for given topic."""
    allfirstwords = []
    with open(topicwordfile, "r") as infile:
        topicwords = pd.read_csv(infile, sep="\t", header=None)
        #print(topicwords)
        topics = topicscores.index.tolist()
        for topic in topics:
            #topic = int(topic[2:])-1
            #topic = int(topic)-1
            topic = int(topic)
            firstwords = topicwords.loc[topic]
            firstwords = firstwords[2].split(" ")
            #topic = topic+1
            firstwords = str(firstwords[0]+"-"+firstwords[1]+"-"+firstwords[2]+" "+str(topic)+"")
            allfirstwords.append(firstwords)   
        #allfirstwords["tp"+str(topic)] = firstwords
        #print(allfirstwords)
        return(allfirstwords)

def create_heatmap(aggregate,topicscores,outfolder,rows_shown,font_scale,dpi):
    """Visualize topic score distribution data as heatmap. """
    data_filename = os.path.basename(aggregate)[:-4]
    print("   Creating heatmap for: "+data_filename)
    ## Create output folder if needed
    outfolder = outfolder+"heatmaps/"
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    sns.set_context("poster", font_scale=font_scale)
    topicscores = topicscores.set_index("firstwords")
    #print(topicscores)
    sns.heatmap(topicscores, annot=False, cmap="YlOrRd", square=False)
    # Nice: bone_r, copper_r, PuBu, OrRd, GnBu, BuGn, YlOrRd
    plt.title("Distribution of topic scores", fontsize=24)
    #plt.xlabel("Categories", fontsize=16)
    plt.ylabel("Top topics (sorted by stdev)", fontsize=20)
    plt.setp(plt.xticks()[1], rotation=90, fontsize = 10)   
    figure_filename = outfolder+"hm_"+ data_filename + ".png"
    plt.tight_layout() 
    plt.savefig(figure_filename, dpi=dpi)
    plt.close()


def create_lineplot(aggregate,topicscores,outfolder,rows_shown,font_scale,dpi,topics,kind): 
    """Visualize topic score distribution data as lineplot. """
    data_filename = os.path.basename(aggregate)[:-4]
    print("   Creating lineplot for: "+data_filename)
    ## Create output folder if needed
    outfolder = outfolder+"lineplots/"
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    ## Create subset of data based on topics to be shown.
    selected = topicscores.loc[topics,:]
    selected = selected.set_index("firstwords")
    selected = selected.T
    #print(selected)
    ## Plot the selected data
    selected.plot(kind=kind) # "line" or "area"
    figure_filename = outfolder+"lp_"+str(topics)+".png"
    plt.savefig(figure_filename, dpi=dpi)
    plt.close()
  

# TODO: Add overall topic score for sorting by overall importance.



def create_topicscores_lineplot(inpath,outfolder,topicwordfile,dpi,height,genres):
    """Generate topic score lineplots from CSV data."""
    print("\nLaunched create_topicscores_lineplot.")

    import os
    import glob
    import re
    import pandas as pd
    import matplotlib.pyplot as plt
    
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    for file in glob.glob(inpath):
        topicscores = pd.DataFrame.from_csv(file, sep="\t")
        #print(topicscores.head())
        topicscores = topicscores.T
        #print(topicscores)
        tpids = topicscores.index
        #print(tpids)
        stdevs = topicscores.std(axis=1)
        topicscores = pd.concat([topicscores, stdevs], axis=1)
        topicscores = topicscores.sort(columns=0, axis=0, ascending=False)
        topicscores = topicscores.iloc[:,0:15]
        # 0:5 = com, 5:10 = trag
        #print(topicscores.iloc[0:2,:]) #rows,columns (but here only 2 columns)

        with open(topicwordfile, "r") as wordfile:
            topics_and_words = wordfile.read()
            topics_and_words = re.split("\n", topics_and_words)
            topicids = []
            fourwords = []
            for topic_and_word in topics_and_words[0:-1]:
                #print(topic_and_word)
                topic_and_word = re.sub("\t.*\t", ",", topic_and_word)
                topicid = re.findall("\d*", topic_and_word)
                topicid = topicid[0]
                topicid = str(topicid)
                topicid = int(topicid)
                topicid = "tp"+"{:03d}".format(topicid)
                #print(topicid)
                topicids.append(topicid)
                fourword = re.sub("[\d]{1,3},([^$]*?[ ])([^$]*?[ ])([^$]*?[ ])([^$]*?[ ])[^$]*", "\\1\\2\\3\\4", topic_and_word, re.DOTALL)
                #print(fourword)
                fourwords.append(fourword)
            topicid_sr = pd.Series(topicids)
            fourword_sr = pd.Series(index=topicid_sr, data=fourwords, name="fourwords")
            #print(fourword_sr)
            #print(fourword_sr["tp000"])

        for tpid in tpids:
            ### Get and plot scores for genre A
            topicscoresA = topicscores.iloc[:,0:5]
            scores = topicscoresA.loc[tpid,]
            plt.plot(scores, lw=4, marker="o", color="red", label=genres[0])

            ### Get and plot scores for genre B
            topicscoresB = topicscores.iloc[:,10:15]
            scores = topicscoresB.loc[tpid,]
            plt.plot(scores, lw=4, color="blue", marker="o", label=genres[1])

            ### Rest of the plot
            plt.title("Distribution over topic scores \n(" + tpid + ")")
            plt.xlabel("Five parts (beginning to end)")
            plt.ylabel("Topic weight")
            leg = plt.legend(frameon=True)
            leg.get_frame().set_edgecolor('grey')
            plt.ylim((0.000,height))
            plt.xlim((0,4))
            tick_locs = [0,1,2,3,4]
            tick_lbls = ["sect.1","sect.2","sect.3","sect.4","sect.5"]
            plt.xticks(tick_locs, tick_lbls)            
            
            plt.grid()
            heightindicator = "{:02d}".format(int(height*100))
            #plt.show()

            ### Save figure
            figure_filename = outfolder + "seg-lp_"+ str(heightindicator) +"_" + tpid + ".png"
            plt.savefig(figure_filename, dpi=dpi)
            plt.close()
            
    print("Done.")

# TODO: find categories automatically and produce graphs for all without "genre" setting.
# TODO: Make plots with lines for all categories in one plot.

