#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: visualize.py
# Authors: christofs, daschloer
# Version 0.3.0 (2016-03-20)

##################################################################
###  Topic Modeling Workflow (tmw)                             ###
##################################################################

##################################################################
### visualize.py -  Visualizations for the model               ###
##################################################################

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import seaborn as sns


#################################
# make_wordle_from_mallet       #
#################################

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
        
def get_color_scale(word, font_size, position, orientation, font_path, random_state=None):
    """ Create color scheme for wordle."""
    return "hsl(245, 58%, 25%)" # Default. Uniform dark blue.
    #return "hsl(0, 00%, %d%%)" % random.randint(80, 100) # Greys for black background.
    #return "hsl(221, 65%%, %d%%)" % random.randint(30, 35) # Dark blues for white background

def get_topicRank(topic, TopicRanksFile):
    #print("getting topic rank.")
    with open(TopicRanksFile, "r") as infile:
        topicRanks = pd.read_csv(infile, sep=",", index_col=0)
        rank = int(topicRanks.iloc[topic]["Rank"])
        return rank

def make_wordle_from_mallet(word_weights_file, 
                            num_topics, 
                            words,
                            TopicRanksFile,
                            outfolder,
                            font_path, 
                            dpi):
    """
    # Generate wordles from Mallet output, using the wordcloud module.
    """
    print("\nLaunched make_wordle_from_mallet.")
    for topic in range(0,num_topics):
        ## Gets the text for one topic.
        text = get_wordlewords(words, word_weights_file, topic)
        wordcloud = WordCloud(font_path=font_path, width=1600, height=1200, background_color="white", margin=4).generate(text)
        default_colors = wordcloud.to_array()
        rank = get_topicRank(topic, TopicRanksFile)
        figure_title = "topic "+ str(topic) + " ("+str(rank)+"/"+str(num_topics)+")"       
        plt.imshow(wordcloud.recolor(color_func=get_color_scale, random_state=3))
        plt.imshow(default_colors)
        plt.imshow(wordcloud)
        plt.title(figure_title, fontsize=30)
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
     
def get_dataToPlot(average, firstWordsFile, mode, topTopicsShown, item):
    """From average topic score data, select data to be plotted."""
    #print("  Getting dataToPlot.")
    with open(average, "r") as infile:
        ## Read the average topic score data
        allData = pd.DataFrame.from_csv(infile, sep=",")
        if mode == "normalized": # mean normalization
            colmeans = allData.mean(axis=0)
            allData = allData / colmeans
        elif mode == "zscores": # zscore transformation
            colmeans = allData.mean(axis=0) # ???
            colstd = allData.std(axis=0) #std for each topic
            allData = (allData - colmeans) / colstd # = zscore transf.
        elif mode == "absolute": # absolute values
            allData = allData
        allData = allData.T
        print(allData)
        ## Add top topic words to table for display later
        firstWords = get_firstWords(firstWordsFile)
        print(len(firstWords))
        allData["firstWords"] = firstWords.iloc[:,0].values
        ## Create subset of data based on target.
        dataToPlot = allData[[item,"firstWords"]]
        dataToPlot = dataToPlot.sort(columns=item, ascending=False)
        dataToPlot = dataToPlot[0:topTopicsShown]
        dataToPlot = dataToPlot.set_index("firstWords")
        #print(dataToPlot)         
        return dataToPlot

def create_barchart_topTopics(dataToPlot, targetCategory, mode, item, 
                              fontscale, height, dpi, outfolder):
    """Function to make a topTopics barchart."""
    print("  Creating plot for: "+str(item))
    ## Doing the plotting.
    dataToPlot.plot(kind="bar", legend=None) 
    plt.setp(plt.xticks()[1], rotation=90, fontsize = 11)   
    if mode == "normalized": 
        plt.title("Top-distinctive Topics für: "+str(item), fontsize=15)
        plt.ylabel("normalized scores", fontsize=13)
    elif mode == "absolute":
        plt.title("Top-wichtigste Topics für: "+str(item), fontsize=15)
        plt.ylabel("absolute scores", fontsize=13)
    plt.xlabel("Topics", fontsize=13)
    plt.tight_layout() 
    if height != 0:
        plt.ylim((0.000,height))
   
    ## Saving the plot to disk.
    outfolder = outfolder+targetCategory+"/"
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    figure_filename = outfolder+"tT_"+mode+"-"+str(item)+".png"
    plt.savefig(figure_filename, dpi=dpi)
    plt.close()

def plot_topTopics(averageDatasets, firstWordsFile, numberOfTopics, 
                   targetCategories, mode, topTopicsShown, fontscale, 
                   height, dpi, outfolder): 
    """For each item in a category, plot the top n topics as a barchart."""
    print("Launched plot_topTopics.")
    for average in glob.glob(averageDatasets):
        for targetCategory in targetCategories: 
            if targetCategory in average:
                targetItems = get_targetItems(average, targetCategory)
                for item in targetItems:
                    dataToPlot = get_dataToPlot(average, firstWordsFile, mode, topTopicsShown, item)
                    create_barchart_topTopics(dataToPlot, targetCategory, mode, item, fontscale, height, dpi, outfolder)
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
                  numOfTopics, 
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
                topics = list(range(0,numOfTopics))
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

def get_heatmap_dataToPlot(average, mode, sorting, firstWordsFile, topTopicsShown, 
                           numOfTopics):
    """From average topic score data, select data to be plotted."""
    print("- getting dataToPlot...")
    with open(average, "r") as infile:
        ## Read the average topic score data
        allScores = pd.DataFrame.from_csv(infile, sep=",")
        colmeans = allScores.mean(axis=0) # mean for each topic
        colmedians = allScores.median(axis=0) # median for each topic
        allstd = allScores.std(axis=0) #std for entire df
        if mode == "meannorm": # mean normalization
            allScores = allScores - colmeans
        if mode == "mediannorm": # mean normalization
            allScores = allScores - colmedians
        if mode == "zscores": # zscore transformation
            allScores = (allScores - colmeans) / allstd # = zscore transf.
        elif mode == "absolute": # absolute values
            allScores = allScores
        allScores = allScores.T
        ## Add top topic words to table for display later
        firstWords = get_heatmap_firstWords(firstWordsFile)
        #allScores = allScores.drop("century")        #TODO: fix underlying issue
        allScores = allScores.drop("decade")         #TODO: fix underlying issue
        allScores = allScores.drop("year")
        allScores = allScores.drop("words")
        allScores = allScores.drop("binID")
        allScores.index = allScores.index.astype(np.int64)        
        allScores = pd.concat([allScores, firstWords], axis=1, join="inner")
        #print(allScores)
        ## Remove undesired columns: subsubgenre
        #allScores = allScores.drop("adventure", axis=1)
        ## Sort by standard deviation
        if sorting == "std":
            standardDeviations = allScores.std(axis=1)
            standardDeviations.name = "std"
            allScores.index = allScores.index.astype(np.int64)        
            allScores = pd.concat([allScores, standardDeviations], axis=1)
            allScores = allScores.sort(columns="std", axis=0, ascending=False)
            allScores = allScores.drop("std", axis=1)
        else: 
            allScores = allScores
        someScores = allScores[0:topTopicsShown]
        #someScores = someScores.drop(0, axis=1)
        ## Necessary step to align dtypes of indexes for concat.
        someScores.index = someScores.index.astype(np.int64)        
        #print("dtype firstWords: ", type(firstWords.index))
        #print("dtype someScores: ", type(someScores.index))
        #print("\n==intersection==\n",someScores.index.intersection(firstWords.index))
        ## Add top topic words to table for display later
        #firstWords = get_heatmap_firstWords(firstWordsFile)
        #print(firstWords)
        dataToPlot = someScores
        dataToPlot = dataToPlot.set_index("topicwords")
        #print(dataToPlot)
        ## Optionally, limit display to part of the columns
        #dataToPlot = dataToPlot.iloc[:,0:40]
        #print(dataToPlot)
        return dataToPlot

def create_distinctiveness_heatmap(dataToPlot, 
                                   topTopicsShown,
                                   targetCategory, 
                                   mode,
                                   sorting,
                                   fontscale,
                                   dpi, 
                                   outfolder):

    sns.set_context("poster", font_scale=fontscale)
    sns.heatmap(dataToPlot, annot=False, cmap=("RdBu_r"), square=False)
    # Nice: bone_r, copper_r, PuBu, OrRd, GnBu, BuGn, YlOrRd, RdBu_r
    plt.title("Topic Score Distribution", fontsize=20)
    plt.xlabel(targetCategory, fontsize=14)
    plt.ylabel("Most distinctive topics", fontsize=14)
    plt.setp(plt.xticks()[1], rotation=0, fontsize = 14)   
    plt.tight_layout() 

    ## Saving the plot to disk.
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    figure_filename = outfolder+"dist-heatmap_by-"+str(targetCategory)+"-"+str(mode)+".png"
    plt.savefig(figure_filename, dpi=dpi)
    plt.close()



def plot_distinctiveness_heatmap(averageDatasets, 
                                 firstWordsFile, 
                                 outfolder, 
                                 targetCategories, 
                                 numOfTopics, 
                                 topTopicsShown, 
                                 mode,
                                 sorting,
                                 fontscale, 
                                 dpi):
    """Visualize topic score distribution data as heatmap. """
    print("Launched plot_distinctiveness_heatmap.")
    for average in glob.glob(averageDatasets):
        for targetCategory in targetCategories: 
            if targetCategory in average and targetCategory != "segmentID":
                print("- working on: "+targetCategory)
                dataToPlot = get_heatmap_dataToPlot(average,
                                                    mode,
                                                    sorting,
                                                    firstWordsFile, 
                                                    topTopicsShown,
                                                    numOfTopics)
                create_distinctiveness_heatmap(dataToPlot, 
                                               topTopicsShown,
                                               targetCategory, 
                                               mode,
                                               sorting,
                                               fontscale,
                                               dpi, 
                                               outfolder)
    print("Done.")


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
## topicClustering     ###
###########################

# TOOD: Add figsize and orientation parameters.
# TODO: Add "firstwords" as leaf labels instead of topic numbers.

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

def perform_topicClustering(scoreMatrix, method, metric, wordsPerTopic, outfolder): 
    print("- performing clustering...")
    distanceMatrix = sc.hierarchy.linkage(scoreMatrix, method=method, metric=metric)
    #print(distanceMatrix)
    plt.figure(figsize=(25,10))
    sc.hierarchy.dendrogram(distanceMatrix)
    plt.setp(plt.xticks()[1], rotation=90, fontsize = 6)   
    plt.title("Topic-Clustering Dendrogramm", fontsize=20)
    plt.ylabel("Distanz", fontsize=16)
    plt.xlabel("Parameter: "+method+" clustering - "+metric+" distance - "+str(wordsPerTopic)+" words", fontsize=16)
    plt.tight_layout() 

    ## Saving the image file.
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    figure_filename = "topic-clustering_"+metric+"-"+method+"-"+str(wordsPerTopic)+"words"+".png"
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
            perform_topicClustering(scoreMatrix, method, metric, wordsPerTopic, outfolder)
    print("Done.")



###########################
## itemClustering       ###
###########################

# TOOD: Add orientation to parameters.

def build_itemScoreMatrix(averageDatasets, targetCategory, 
                          topicsPerItem, sortingCriterium):
    """Reads Mallet output (topics with words and word weights) into dataframe.""" 
    print("- getting topWordScores...")
    for averageFile in glob.glob(averageDatasets): 
        if targetCategory in averageFile:
            itemScores = pd.read_table(averageFile, header=0, index_col=0, sep=",")
            itemScores = itemScores.T 
            if sortingCriterium == "std": 
                itemScores["sorting"] = itemScores.std(axis=1)
            elif sortingCriterium == "mean": 
                itemScores["sorting"] = itemScores.mean(axis=1)
            itemScores = itemScores.sort(columns=["sorting"], axis=0, ascending=False)
            itemScoreMatrix = itemScores.iloc[0:topicsPerItem,0:-1]
            itemScoreMatrix = itemScoreMatrix.T
            #itemScoreMatrix = itemScoreMatrix.drop("Allais", axis=0)
            #print(itemScoreMatrix)
            return itemScoreMatrix

def perform_itemClustering(itemScoreMatrix, targetCategory, method, metric, 
                           topicsPerItem, sortingCriterium, figsize, outfolder): 
    print("- performing clustering...")

    ## Perform the actual clustering
    itemDistanceMatrix = sc.hierarchy.linkage(itemScoreMatrix, method=method, metric=metric)
        
    ## Plot the distance matrix as a dendrogram
    plt.figure(figsize=figsize) # TODO: this could be a a parameter.
    itemLabels = itemScoreMatrix.index.values
    sc.hierarchy.dendrogram(itemDistanceMatrix, labels=itemLabels, orientation="top")

    ## Format items labels to x-axis tick labels
    plt.setp(plt.xticks()[1], rotation=90, fontsize = 14)
    plt.title("Item Clustering Dendrogramm: "+targetCategory, fontsize=20)
    plt.ylabel("Distance", fontsize=16)
    plt.xlabel("Parameter: "+method+" clustering - "+metric+" distance - "+str(topicsPerItem)+" topics", fontsize=16)
    plt.tight_layout() 

    ## Save the image file.
    print("- saving image file.")
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    figure_filename = "item-clustering_"+targetCategory+"_"+metric+"-"+method+"-"+sortingCriterium+"-"+str(topicsPerItem)+"topics"+".jpg"
    plt.savefig(outfolder + figure_filename, dpi=600)
    plt.close()
    
def itemClustering(averageDatasets, figsize, outfolder, topicsPerItem, 
                   targetCategories, methods, metrics, sortingCriterium):
    """Display dendrogram of topic-based item similarity using clustering."""
    print("\nLaunched itemClustering.")
    for targetCategory in targetCategories: 
        ## Load topic scores per itema and turn into score matrix
        itemScoreMatrix = build_itemScoreMatrix(averageDatasets, targetCategory, 
                                                topicsPerItem, sortingCriterium)
        ## Do clustering on the dataframe
        for method in methods: 
            for metric in metrics: 
                perform_itemClustering(itemScoreMatrix, targetCategory, 
                                       method, metric, topicsPerItem, 
                                       sortingCriterium, figsize, outfolder)
    print("Done.")




###########################
## simple progression   ###
###########################


def get_progression_firstWords(firstWordsFile):
    """Function to load list of top topic words into dataframe."""
    #print("  Getting firstWords.")
    with open(firstWordsFile, "r") as infile: 
        firstWords = pd.read_csv(infile, header=None)
        firstWords.drop(0, axis=1, inplace=True)
        firstWords.rename(columns={1:"topicwords"}, inplace=True)
        firstWords.index = firstWords.index.astype(np.int64)        
        #print(firstWords)
        return(firstWords)


def get_selSimpleProgression_dataToPlot(averageDataset, firstWordsFile, 
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
    
    
def create_selSimpleProgression_lineplot(dataToPlot, outfolder, fontscale, 
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

def get_allSimpleProgression_dataToPlot(averageDataset, firstWordsFile, 
                               entriesShown, topic): 
    """Function to build a dataframe with all data necessary for plotting."""
    print("- getting data to plot...")
    with open(averageDataset, "r") as infile:
        allScores = pd.DataFrame.from_csv(infile, sep=",")
        allScores = allScores.T        
        #print(allScores)
        ## Select the data for current topics
        someScores = allScores.loc[topic,:]
        someScores.index = someScores.index.astype(np.int64)
        dataToPlot = someScores
        #print(dataToPlot)
        return dataToPlot
        
# TODO: Make sure this is only read once and then select when plotting.
    
    
def create_allSimpleProgression_lineplot(dataToPlot, outfolder, fontscale, 
                                firstWordsFile, topic, dpi, height):
    """This function does the actual plotting and saving to disk."""
    print("- creating the plot for topic " + topic)
    ## Get the first words info for the topic
    firstWords = get_progression_firstWords(firstWordsFile)
    topicFirstWords = firstWords.iloc[int(topic),0]
    #print(topicFirstWords)
    ## Plot the selected data
    dataToPlot.plot(kind="line", lw=3, marker="o")
    plt.title("Entwicklung über den Textverlauf für "+topicFirstWords, fontsize=20)
    plt.ylabel("Topic scores (absolut)", fontsize=16)
    plt.xlabel("Textabschnitte", fontsize=16)
    plt.setp(plt.xticks()[1], rotation=0, fontsize = 14)   
    if height != 0:
        plt.ylim((0.000,height))

    ## Saving the plot to disk.
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    ## Format the topic information for display
    topicsLabel = str(topic)
    figure_filename = outfolder+"all_"+topicsLabel+".png"
    plt.savefig(figure_filename, dpi=dpi)
    plt.close()


def simpleProgression(averageDataset, firstWordsFile, outfolder, 
                           numOfTopics, 
                           fontscale, dpi, height, mode, topics):
    """Function to plot topic development over textual progression."""
    print("Launched textualProgression.")
    if mode == "selected" or mode == "sel": 
        entriesShown = numOfTopics
        dataToPlot = get_selSimpleProgression_dataToPlot(averageDataset, 
                                                      firstWordsFile, 
                                                      entriesShown, 
                                                      topics)
        create_selSimpleProgression_lineplot(dataToPlot, outfolder, 
                                          fontscale, topics, 
                                          dpi, height)
    elif mode == "all": 
        entriesShown = numOfTopics
        topics = list(range(0, numOfTopics))
        for topic in topics:
            topic = str(topic)
            dataToPlot = get_allSimpleProgression_dataToPlot(averageDataset, 
                                                             firstWordsFile, 
                                                             entriesShown, 
                                                             topic)
            create_allSimpleProgression_lineplot(dataToPlot, outfolder, 
                                                 fontscale, firstWordsFile, 
                                                 topic, dpi, height)
    else: 
        print("Please select a valid value for 'mode'.")
    print("Done.")



#def main(word_weights_file, num_topics, words, TopicRanksFile, outfolder, font_path, dpi):
#    make_wordle_from_mallet(word_weights_file, num_topics, words, TopicRanksFile, outfolder, font_path, dpi)
#    crop_images(inpath, outfolder, left, upper, right, lower)
#    plot_topTopics(averageDatasets, firstWordsFile, numberOfTopics, targetCategories, topTopicsShown, fontscale, height, dpi, outfolder)
#   plot_topItems(averageDatasets, outfolder, firstWordsFile, numberOfTopics, targetCategories, topItemsShown, fontscale, height, dpi)
#    plot_distinctiveness_heatmap(averageDatasets, firstWordsFile, outfolder, targetCategories, numberOfTopics, topTopicsShown, fontscale, dpi)    
#    plot_topicsOverTime(averageDatasets, firstWordsFile, outfolder, numberOfTopics, fontscale, dpi, height, mode, topics)
    
#if __name__ == "__main__":
#    import sys
#    make_wordle_from_mallet(int(sys.argv[1]))
#    crop_images(int(sys.argv[1]))
#    plot_topTopics(int(sys.argv[1]))
#    plot_topItems(int(sys.argv[1]))
#    plot_distinctiveness_heatmap(int(sys.argv[1]))
#    plot_topicsOverTime(int(sys.argv[1]))

