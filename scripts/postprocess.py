#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: postprocess.py
# Authors: christofs, daschloer, hennyu
# Version 0.3.0 (2016-03-20)


"""
POSTPROCESSING OF RAW TMW DATA
"""


import os
from os.path import join
import glob
import pandas as pd
import numpy as np
import re

##############################
# create_mastermatrix        #
##############################

def get_metadata(metadatafile):
    print("- getting metadata...")
    """Read metadata file and create DataFrame."""
    metadata = pd.read_csv(metadatafile, header=0, sep=",")
    #print("metadata\n", metadata)
    return metadata

def get_topicscores(topics_in_texts, numOfTopics, version): 
    """Create a matrix of segments x topics, with topic score values, from Mallet output.""" 
    print("- getting topicscores...")  
    
    if version == "207":
        print("(Warning: With Mallet 2.0.7 output, this is very memory-intensive.)")
        ## Load Mallet 2.0.7 output (strange format)
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
            for j in range(1,numOfTopics,2):
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
        print("topicscores\n", topicscores.head())
        return topicscores

    if version == "208+":
        ## Load Mallet output (new, not so strange format)
        header = list(range(0,numOfTopics)) ## for use as column headers in dataframe
        header = ["segmentID"] + header
        #print(header)
        topicsintexts = pd.read_csv(topics_in_texts, header=None, sep="\t", index_col=0)
        topicsintexts.columns = header
        #print(topicsintexts)
        ##topicsintexts = topicsintexts.iloc[0:100,]  ### For testing only!!
        if "§" in topicsintexts.iloc[0,0]:
            segmentIDs = topicsintexts.iloc[:,0].str[-15:-4]
        else:
            segmentIDs = topicsintexts.iloc[:,0].str[-10:-4]
        topicsintexts["segmentID"] = segmentIDs
        topicscores = topicsintexts
        #print("topicscores\n", topicscores.head())
        return topicscores

        
def get_docmatrix(corpuspath):
    """Create a matrix containing segments with their idnos."""
    print("- getting docmatrix...")
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
               numOfTopics, version):
    """Merges the three dataframes into one mastermatrix."""
    print("- getting data...")
    ## Get all necessary data.
    metadata = get_metadata(metadatafile)
    docmatrix = get_docmatrix(corpuspath)
    topicscores = get_topicscores(topics_in_texts, numOfTopics, version)
    ## For inspection only.
    ##print("Metadata\n", metadata.head())
    ##print("Docmatrix\n", docmatrix.head())
    ##print("topicscores\n", topicscores.head())
    
    print("- merging data...")    
    ## Merge metadata and docmatrix, matching each segment to its metadata.
    
    if not("idno" in metadata.columns):
        metadata["idno"] = metadata.index
    
    mastermatrix = pd.merge(docmatrix, metadata, how="inner", on="idno")  
    #print("mastermatrix: metadata and docmatrix\n", mastermatrix)
    ## Merge mastermatrix and topicscores, matching each segment to its topic scores.
    #print(mastermatrix.columns)
    #print(topicscores.columns)
    #print(topicscores)
    mastermatrix = pd.merge(mastermatrix, topicscores, on="segmentID", how="inner")
    #print("mastermatrix: all three\n", mastermatrix.head())
    return mastermatrix

def add_binData(mastermatrix, binDataFile): 
    print("- adding bin data...")
    ## Read the information about bins
    binData = pd.read_csv(binDataFile, sep=",")
    #print(binData)
    ## Merge existing mastermatrix and binData.
    mastermatrix = pd.merge(mastermatrix, binData, how="inner", on="segmentID")  
    #print(mastermatrix)
    return mastermatrix

def create_mastermatrix(corpuspath, outfolder, mastermatrixfile, metadatafile, 
                        topics_in_texts, numOfTopics, useBins, binDataFile, version):
    """Builds the mastermatrix uniting all information about texts and topic scores."""
    print("\nLaunched create_mastermatrix.")
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    mastermatrix = merge_data(corpuspath, metadatafile, topics_in_texts, 
                              mastermatrixfile, numOfTopics, version)
    if useBins == True: 
        mastermatrix = add_binData(mastermatrix, binDataFile)
    mastermatrix.to_csv(join(outfolder, mastermatrixfile), sep=",", encoding="utf-8")
    print("Done. Saved mastermatrix. Segments and columns:", mastermatrix.shape)    



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
        
        for col in avg_topicscores.columns:
            if not(re.match("\d+", col)):
                avg_topicscores = avg_topicscores.drop([col], axis=1)
        
        #avg_topicscores = grouped.agg(np.median)
        #print(avg_topicscores)
        
        #if target != "pub-year":
        #    avg_topicscores = avg_topicscores.drop(["pub-year"], axis=1)
        
        #if target != "binID":
        #    avg_topicscores = avg_topicscores.drop(["binID"], axis=1)
        #avg_topicscores = avg_topicscores.drop(["tei"], axis=1)
        
        ## Save grouped averages to CSV file for visualization.
        resultfilename = "avgtopicscores_by-"+target+".csv"
        resultfilepath = join(outfolder, resultfilename)
        ## TODO: Some reformatting here, or adapt make_heatmaps.
        avg_topicscores.to_csv(resultfilepath, sep=",", encoding="utf-8")
        print("  Saved average topic scores for:", target)    
    print("Done.")




################################
# complexAverageTopicScores    #
################################

def calculate_complexAverageTopicScores(mastermatrixfile, targets, outfolder):
    """Function to calculate average topic scores based on the mastermatrix."""
    print("\nLaunched calculate_complexAverageTopicScores.")
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    with open(mastermatrixfile, "r", encoding="utf-8") as infile:
        mastermatrix = pd.DataFrame.from_csv(infile, header=0, sep=",")
    ## Calculate average topic scores for each target category 
    grouped = mastermatrix.groupby(targets, axis=0)
    avg_topicscores = grouped.agg(np.mean)
    
    for col in avg_topicscores.columns:
        if not(re.match("\d+", col)):
            avg_topicscores = avg_topicscores.drop([col], axis=1)
    
    #if "year" not in targets:
    #    avg_topicscores = avg_topicscores.drop(["year"], axis=1)
    #if "binID" not in targets:
    #    avg_topicscores = avg_topicscores.drop(["binID"], axis=1)
    #print(avg_topicscores)
    ## Save grouped averages to CSV file for visualization.
    identifierstring = '+'.join(map(str, targets))
    resultfilename = "complex-avgtopicscores_by-"+identifierstring+".csv"
    resultfilepath = join(outfolder, resultfilename)
    avg_topicscores.to_csv(resultfilepath, sep=",", encoding="utf-8")
    print("Done. Saved average topic scores for: "+identifierstring)    



#################################
# save_firstWords               #
#################################

def save_firstWords(topicWordFile, outfolder, filename):
    """Save a table of topics with their three most important words for each topic."""
    print("Launched save_someFirstWords.")
    with open(topicWordFile, "r", encoding="utf-8") as infile:
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
        #print(firstWordsSeries)
        ## Saving the file.
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
        outfile = join(outfolder, filename)
        with open(outfile, "w", encoding="utf-8") as outfile: 
            firstWordsSeries.to_csv(outfile)
        print("Done.")


#################################
# save_topicRanks               #
#################################

def save_topicRanks(topicWordFile, outfolder, filename):
    """Save a list of topics with their rank by topic score."""
    print("Launched save_topicRanks.")
    with open(topicWordFile, "r", encoding="utf-8") as infile:
        topicRanks = pd.read_csv(infile, sep="\t", header=None)
        topicRanks = topicRanks.drop(2, axis=1)
        topicRanks.rename(columns={0:"Number"}, inplace=True)
        topicRanks.rename(columns={1:"Score"}, inplace=True)
        #topicRanks.sort(columns=["Score"], ascending=False, inplace=True)
        topicRanks["Rank"] = topicRanks["Score"].rank(ascending=False)
        #print(topicRanks.head())
        ## Saving the file.
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
        outfile = join(outfolder, filename)
        with open(outfile, "w", encoding="utf-8") as outfile: 
            topicRanks.to_csv(outfile)
        print("Done.")

