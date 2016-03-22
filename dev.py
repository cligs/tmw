#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: dev.py
# Authors: christofs, daschloer
# Version 0.3.0 (2016-03-20)

##################################################################
###  Topic Modeling Workflow (tmw)                             ###
##################################################################


##################################################################
### dev.py - OTHER / OBSOLETE / DEV                            ###
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


###########################
## complex progression  ###        IN DEVELOPMENT
###########################


def get_selComplexProgression_dataToPlot(averageDataset, firstWordsFile, 
                               entriesShown, topics): 
    """Function to build a dataframe with all data necessary for plotting."""
    print("- getting data to plot...")
    with open(averageDataset, "r") as infile:
        allScores = pd.DataFrame.from_csv(infile, sep=",")
        allScores = allScores.T        
        #print(allScores.head())
        ## Select the data for selected topics
        someScores = allScores.loc[topics,:]
        someScores.index = someScores.index.astype(np.int64)        
        ## Add information about the firstWords of topics
        firstWords = get_progression_firstWords(firstWordsFile)
        dataToPlot = pd.concat([someScores, firstWords], axis=1, join="inner")
        dataToPlot = dataToPlot.set_index("topicwords")
        dataToPlot = dataToPlot.T
        #print(dataToPlot)
        return dataToPlot
    
    
def create_selComplexProgression_lineplot(dataToPlot, outfolder, fontscale, 
                                topics, dpi, height):
    """This function does the actual plotting and saving to disk."""
    print("- creating the plot...")
    ## Plot the selected data
    dataToPlot.plot(kind="line", lw=3, marker="o")
    plt.title("Entwicklung ausgewählter Topics über den Textverlauf", fontsize=20)
    plt.ylabel("Topic scores (absolut)", fontsize=16)
    plt.xlabel("Textabschnitte", fontsize=16)
    plt.setp(plt.xticks()[1], rotation=0, fontsize = 14)   
    if height != 0:
        plt.ylim((0.000,height))

    ## Saving the plot to disk.
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    ## Format the topic information for display
    topicsLabel = "-".join(str(topic) for topic in topics)
    figure_filename = outfolder+"sel_"+topicsLabel+".png"
    plt.savefig(figure_filename, dpi=dpi)
    plt.close()

def get_allComplexProgression_dataToPlot(averageDataset, firstWordsFile, 
                                         entriesShown, topic, targetCategories): 
    """Function to build a dataframe with all data necessary for plotting."""
    print("- getting data to plot...")
    with open(averageDataset, "r") as infile:
        allScores = pd.DataFrame.from_csv(infile, sep=",", index_col=None)
        #print(allScores)
        ## Select the data for current topics
        target1 = targetCategories[0]
        target2 = targetCategories[1]
        target1data = allScores.loc[:,target1]
        target2data = allScores.loc[:,target2]
        topicScores = allScores.loc[:,topic]
        #print(target1data)
        #print(target2data)
        #print(topicScores)
        dataToPlot = pd.concat([target1data, target2data], axis=1)
        dataToPlot = pd.concat([dataToPlot, topicScores], axis=1)
        #print(dataToPlot)
        return dataToPlot
        
# TODO: Make sure this is only read once and then select when plotting.

        
def create_allComplexProgression_lineplot(dataToPlot, targetCategories, 
                                          outfolder, fontscale, 
                                firstWordsFile, topic, dpi, height):
    """This function does the actual plotting and saving to disk."""
    print("- creating the plot for topic " + topic)
    ## Get the first words info for the topic
    firstWords = get_progression_firstWords(firstWordsFile)
    topicFirstWords = firstWords.iloc[int(topic),0]
    #print(topicFirstWords)
    ## Split plotting data into parts (for target1)
    target1data = dataToPlot.iloc[:,0]
    #print(target1data)
    numPartialData = len(set(target1data))
    ## Initialize plot for several lines
    completeData = []
    #print(dataToPlot)
    for target in set(target1data):
        #print("  - plotting "+target)
        partialData = dataToPlot.groupby(targetCategories[0])
        partialData = partialData.get_group(target)
        partialData.rename(columns={topic:target}, inplace=True)
        partialData = partialData.iloc[:,2:3]
        completeData.append(partialData)
    #print(completeData)
    ## Plot the selected data, one after the other
    plt.figure()
    plt.figure(figsize=(15,10))
    for i in range(0, numPartialData):
        #print(completeData[i])
        label = completeData[i].columns.values.tolist()
        label = str(label[0])
        plt.plot(completeData[i], lw=4, marker="o", label=label)
        plt.legend()
    plt.title("Entwicklung über den Textverlauf für "+topicFirstWords, fontsize=20)
    plt.ylabel("Topic scores (absolut)", fontsize=16)
    plt.xlabel("Textabschnitte", fontsize=16)
    plt.legend()
    plt.locator_params(axis = 'x', nbins = 10)
    plt.setp(plt.xticks()[1], rotation=0, fontsize = 14)   
    if height != 0:
        plt.ylim((0.000,height))

    ## Saving the plot to disk.
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    ## Format the topic information for display
    topicsLabel = str(topic)
    figure_filename = outfolder+"all_"+str(targetCategories[0])+"-"+topicsLabel+".png"
    plt.savefig(figure_filename, dpi=dpi)
    plt.close()


def complexProgression(averageDataset, 
                       firstWordsFile, 
                       outfolder, 
                       numOfTopics, 
                       targetCategories, 
                       fontscale, 
                       dpi, height, 
                       mode, topics):
    """Function to plot topic development over textual progression."""
    print("Launched complexProgression.")
    if mode == "sel": 
        entriesShown = numOfTopics
        dataToPlot = get_selSimpleProgression_dataToPlot(averageDataset, 
                                                         firstWordsFile, 
                                                         entriesShown, 
                                                         topics)
        create_selSimpleProgression_lineplot(dataToPlot, 
                                             outfolder, 
                                             fontscale, 
                                             topics, 
                                             dpi, height)
    elif mode == "all": 
        entriesShown = numOfTopics
        topics = list(range(0, numOfTopics))
        for topic in topics:
            topic = str(topic)
            dataToPlot = get_allComplexProgression_dataToPlot(averageDataset, 
                                                             firstWordsFile, 
                                                             entriesShown, 
                                                             topic,
                                                             targetCategories)
            create_allComplexProgression_lineplot(dataToPlot, targetCategories,
                                                  outfolder, 
                                                  fontscale, firstWordsFile, 
                                                  topic, dpi, height)
    else: 
        print("Please select a valid value for 'mode'.")
    print("Done.")
    
    


###########################
## show_segment         ###
###########################

import shutil

def show_segment(wdir,segmentID, outfolder): 
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    shutil.copyfile(wdir+"2_segs/"+segmentID+".txt",outfolder+segmentID+".txt")




###########################
## itemPCA              ###            IN DEVELOPMENT
###########################

from sklearn.decomposition import PCA

#def build_itemScoreMatrix(averageDatasets, targetCategory, 
#                          topicsPerItem, sortingCriterium):
#    """Reads Mallet output (topics with words and word weights) into dataframe.""" 
#    print("- building item score matrix...")
#    for averageFile in glob.glob(averageDatasets): 
#        if targetCategory in averageFile:
#            itemScores = pd.read_table(averageFile, header=0, index_col=0, sep=",")
#            itemScores = itemScores.T 
#            if sortingCriterium == "std": 
#                itemScores["sorting"] = itemScores.std(axis=1)
#            elif sortingCriterium == "mean": 
#                itemScores["sorting"] = itemScores.mean(axis=1)
#            itemScores = itemScores.sort(columns=["sorting"], axis=0, ascending=False)
#            itemScoreMatrix = itemScores.iloc[0:topicsPerItem,0:-1]
#            itemScoreMatrix = itemScoreMatrix.T
#            #print(itemScoreMatrix)
#            return itemScoreMatrix

def perform_itemPCA(itemScoreMatrix, targetCategory, topicsPerItem, 
                    sortingCriterium, figsize, outfolder):
    print("- doing the PCA...")
    itemScoreMatrix = itemScoreMatrix.T
    targetDimensions = 2
    pca = PCA(n_components=targetDimensions)
    pca = pca.fit(itemScoreMatrix)
    pca = pca.transform(itemScoreMatrix)
#   plt.scatter(pca[0,0:20], pca[1,0:20])
    for i in list(range(0,len(pca)-1)):
        plt.scatter(pca[i,:], pca[i+1,:])


def itemPCA(averageDatasets, targetCategories, 
            topicsPerItem, sortingCriterium, figsize, outfolder): 
    """Function to perform PCA on per-item topic scores and plot the result."""
    print("Launched itemPCA.")
    for targetCategory in targetCategories: 
        ## Load topic scores per item and turn into score matrix
        ## (Using the function from itemClustering above!)
        itemScoreMatrix = build_itemScoreMatrix(averageDatasets, targetCategory, 
                                            topicsPerItem, sortingCriterium)
        ## Do clustering on the dataframe
        perform_itemPCA(itemScoreMatrix, targetCategory, topicsPerItem, sortingCriterium, figsize, outfolder)
    print("Done.")

    
    

    
