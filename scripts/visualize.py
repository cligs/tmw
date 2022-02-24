#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: visualize.py
# Authors: christofs, daschloer, hennyu
# Version 0.3.0 (2016-03-20)
# Update: 2017-10-22

##################################################################
###  Topic Modeling Workflow (tmw)                             ###
##################################################################

##################################################################
### visualize.py -  Visualizations for the model               ###
##################################################################

import os
from os.path import join
import glob
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
import numpy as np
import seaborn as sns
import pygal


#################################
# make_wordle_from_mallet       #
#################################

def read_mallet_output(word_weights_file):
    """Reads Mallet output (topics with words and word weights) into dataframe.""" 
    word_scores = pd.read_csv(word_weights_file, header=None, sep="\t")
    word_scores = word_scores.sort_values(by=[0,2], axis=0, ascending=[True, False])
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
        score = word_scores[j]
        j += 1
        wordlewords = wordlewords + ((word + " ") * int(score))
    return wordlewords
        
def get_color_scale(word, font_size, position, orientation, font_path, random_state=None):
    """ Create color scheme for wordle."""
    return "hsl(245, 58%, 25%)" # Default. Uniform dark blue.
    #return "hsl(0, 00%, %d%%)" % random.randint(80, 100) # Greys for black background.
    #return "hsl(221, 65%%, %d%%)" % random.randint(30, 35) # Dark blues for white background

def get_topicRank(topic, TopicRanksFile):
    #print("getting topic rank.")
    with open(TopicRanksFile, "r", encoding="utf-8") as infile:
        topicRanks = pd.read_csv(infile, sep=",", index_col=0)
        rank = int(topicRanks.iloc[topic]["Rank"])
        return rank

def make_wordle_from_mallet(word_weights_file, 
                            num_topics, 
                            words,
                            TopicRanksFile,
                            outfolder,
                            #font_path, 
                            dpi):
    """
    # Generate wordles from Mallet output, using the wordcloud module.
    """
    print("\nLaunched make_wordle_from_mallet.")
    for topic in range(0,num_topics):
        ## Gets the text for one topic.
        text = get_wordlewords(words, word_weights_file, topic)
        wordcloud = WordCloud(width=1600, height=1200, background_color="white", margin=4, collocations=False).generate(text) #font_path=font_path, 
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
        plt.savefig(join(outfolder, figure_filename), dpi=dpi)
        plt.close()
    print("Done.")
    
def crop_images(inpath, outfolder, left, upper, right, lower):
    """ Function to crop wordle files."""
    print("Launched crop_images.")

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

# TODO: Move this one level up if several plotting functions use it.
def get_firstWords(firstWordsFile):
    """Function to load list of top topic words into dataframe."""
    #print("  Getting firstWords.")
    with open(firstWordsFile, "r", encoding="utf-8") as infile: 
        firstWords = pd.read_csv(infile, header=None)
        firstWords.drop(0, axis=1, inplace=True)
        firstWords.rename(columns={1:"topicwords"}, inplace=True)
        #print(firstWords)
        return(firstWords)

def get_targetItems(average, targetCategory):
    """Get a list of items included in the target category."""
    print(" Getting targetItems for: "+targetCategory)
    with open(average, "r", encoding="utf-8") as infile:
        averageTopicScores = pd.DataFrame.from_csv(infile, sep=",")
        #print(averageTopicScores.head())
        targetItems = list(averageTopicScores.index.values)
        #print(targetItems)
        return(targetItems)    
     
def get_dataToPlot(average, firstWordsFile, mode, topTopicsShown, item):
    """From average topic score data, select data to be plotted."""
    #print("  Getting dataToPlot.")
    with open(average, "r", encoding="utf-8") as infile:
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
        
        '''
        allData = allData.drop("year")
        allData = allData.drop("words")
        '''
        print(allData.shape)
        ## Add top topic words to table for display later
        firstWords = get_firstWords(firstWordsFile)
        print(len(firstWords))
        allData["firstWords"] = firstWords.iloc[:,0].values
        ## Create subset of data based on target.
        dataToPlot = allData[[item,"firstWords"]]
        dataToPlot = dataToPlot.sort_values(by=item, ascending=False)
        dataToPlot = dataToPlot[0:topTopicsShown]
        dataToPlot = dataToPlot.set_index("firstWords")
        #print(dataToPlot)         
        return dataToPlot

def create_barchart_topTopics(dataToPlot, targetCategory, mode, item, 
                              fontscale, height, dpi, outfolder):
    """Function to make a topTopics barchart."""
    print("  Creating plot for: "+str(item))
    ## Doing the plotting.
    dataToPlot.plot(kind="bar", legend=None, color="#003399") 
    plt.setp(plt.xticks()[1], rotation=90, fontsize = 11)   
    if mode == "normalized": 
        plt.title("Top distinctive topics for: "+str(item), fontsize=15)
        plt.ylabel("normalized scores", fontsize=13)
    elif mode == "absolute":
        plt.title("Top-wichtigste Topics für: "+str(item), fontsize=15)
        plt.ylabel("absolute scores", fontsize=13)
    plt.xlabel("Topics", fontsize=13)
    plt.tight_layout() 
    if height != 0:
        plt.ylim((0.000,height))
   
    ## Saving the plot to disk.
    outfolder = join(outfolder, targetCategory)
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    figure_filename = join(outfolder, "tT_"+mode+"-"+str(item)+".png")
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
    with open(firstWordsFile, "r", encoding="utf-8") as infile: 
        firstWords = pd.DataFrame.from_csv(infile, header=None)
        firstWords.columns = ["firstWords"]
        # Only the words for one topic are needed.
        firstWords = firstWords.iloc[topic]
        firstWords = firstWords[0]
        return(firstWords)

def get_topItems_dataToPlot(average, firstWordsFile, topItemsShown, topic):
    """From average topic score data, select data to be plotted."""
    #print("  Getting dataToPlot.")
    with open(average, "r", encoding="utf-8") as infile:
        ## Read the average topic score data
        allData = pd.DataFrame.from_csv(infile, sep=",")
        allData = allData.T
        ## Create subset of data based on target.
        dataToPlot = allData.iloc[topic,:]
        dataToPlot = dataToPlot.sort_values(ascending=False)
        dataToPlot = dataToPlot[0:topItemsShown]
        #print(dataToPlot)
        return dataToPlot

def create_topItems_barchart(dataToPlot, firstWords, targetCategory, topic, 
                              fontscale, height, dpi, outfolder):
    """Function to make a topItems barchart."""
    print("  Creating plot for topic: "+str(topic))
    ## Doing the plotting.
    dataToPlot.plot(kind="bar", legend=None, color="#003399") 
    plt.title("Top "+targetCategory+" für topic: "+str(firstWords), fontsize=15)
    plt.ylabel("Scores", fontsize=13)
    plt.xlabel(targetCategory, fontsize=13)
    plt.setp(plt.xticks()[1], rotation=90, fontsize = 11)   
    if height != 0:
        plt.ylim((0.000,height))
    plt.tight_layout() 

    ## Saving the plot to disk.
    outfolder = join(outfolder, targetCategory)
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    figure_filename = join(outfolder, "tI_by-"+targetCategory+"-{:03d}".format(topic)+".png")
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
    with open(firstWordsFile, "r", encoding="utf-8") as infile: 
        firstWords = pd.read_csv(infile, header=None)
        firstWords.drop(0, axis=1, inplace=True)
        firstWords.rename(columns={1:"topicwords"}, inplace=True)
        #print(firstWords)
        return(firstWords)

def get_heatmap_dataToPlot(average, mode, sorting, firstWordsFile, topTopicsShown, 
                           numOfTopics):
    """From average topic score data, select data to be plotted."""
    print("- getting dataToPlot...")
    with open(average, "r", encoding="utf-8") as infile:
        ## Read the average topic score data
        allScores = pd.DataFrame.from_csv(infile, sep=",")
        colmeans = allScores.mean(axis=0) # mean for each topic
        colmedians = allScores.median(axis=0) # median for each topic
        allstd = allScores.std(axis=0) #std for entire df
        if mode == "meannorm": # mean normalization
            allScores = allScores - colmeans
        if mode == "mediannorm": # median normalization
            allScores = allScores - colmedians
        if mode == "zscores": # zscore transformation
            allScores = (allScores - colmeans) / allstd # = zscore transf.
        elif mode == "absolute": # absolute values
            allScores = allScores
        allScores = allScores.T
        ## Add top topic words to table for display later
        firstWords = get_heatmap_firstWords(firstWordsFile)
        '''
        allScores = allScores.drop("century")        #TODO: fix underlying issue
        allScores = allScores.drop("year")
        allScores = allScores.drop("words")
        '''
        
        
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
            allScores = allScores.sort_values(by="std", axis=0, ascending=False)
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
    plt.setp(plt.xticks()[1], rotation=90, fontsize = 14)   
    plt.setp(plt.yticks()[1], rotation=0, fontsize = 14)
    plt.tight_layout() 

    ## Saving the plot to disk.
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    figure_filename = join(outfolder, "dist-heatmap_by-"+str(targetCategory)+"-"+str(mode)+".png")
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
# plot_words_in_topics_treemap  #
#################################


def plot_words_in_topics_treemap(num_topics, words_to_plot, word_weights_file, wordsintopics_treemap_out):
    """
    author: hennyu
    
    Arguments:
    num_topics (int): number of topics
    words_to_plot (int): how many words to consider for each topic
    word_weights_file (str): path to the word weights file
    wordsintopics_treemap_out (str): output directory for the treemaps
    """
    
    print("\nLaunched plot_words_in_topics_treemap.")
    
    """
    Check directory.
    """
    if not(os.path.exists(wordsintopics_treemap_out)):
        os.makedirs(wordsintopics_treemap_out)
    
    word_weights = pd.read_csv(word_weights_file, sep="\t", encoding="utf-8", header=None)
    
    for topic in word_weights.iloc[:,0].unique():
		
        topic_words = word_weights.loc[word_weights.iloc[:,0] == topic].sort_values(by=2,ascending=False)
        top_words = topic_words[:words_to_plot]
        
        """
        Plot treemaps of how the top words are distributed in each topic.
        """
        treemap = pygal.Treemap(print_values=True, print_labels=True)
        treemap.title = 'Words-in-topics treemap for topic ' + str(topic)
        
        for idx_word, word in top_words.iterrows():
            treemap.add(word.iloc[1], [{"label" : word.iloc[1], "value": round(word.iloc[2])}])
            
        treemap.render_to_file(join(wordsintopics_treemap_out, "treemap_tp" + str(topic) + ".svg"))
    
    print("Done.")
    
	


#####################################
# plot_topics_in_docs_treemap  #
#####################################

def plot_topics_in_docs_treemap(topics_to_plot, doc_topic_file, first_words_file, topicsindocs_treemap_out):
    """
    author: hennyu
    
    Arguments:
    topics_to_plot (int): how many of the top topics to plot for a document
    doc_topic_file (str): path to the doc topic matrix file
    first_words_file (str): path to the first words file
    topicsindocs_treemap_out (str): output directory for the treemaps
    """
    
    print("\nLaunched plot_topics_in_docs_treemap.")
    
    """
    Check directory.
    """
    if not(os.path.exists(topicsindocs_treemap_out)):
        os.makedirs(topicsindocs_treemap_out)

    doc_topics = pd.read_csv(doc_topic_file, sep=",", encoding="utf-8", header=0)
    first_words = pd.read_csv(first_words_file, sep=",", encoding="utf-8", header=None)
    
    for doc in doc_topics.iterrows():
        
        doc_id = doc[1].values[0]
        doc_topics = doc[1].values[1:]
        top_topics = sorted(range(len(doc_topics)), key=lambda i: doc_topics[i])[-topics_to_plot:]
        
        """
        Plot treemaps of how the top words are distributed in each topic.
        """
        treemap = pygal.Treemap(print_values=True, print_labels=True)
        treemap.title = 'Topics-in-docs treemap for document ' + str(doc_id)
        
        for topic in top_topics:
            treemap.add(str(topic), [{"label" : str(first_words.iloc[topic].values[1]), "value": float("%.2f" % doc_topics[topic])}])
            
        treemap.render_to_file(join(topicsindocs_treemap_out, "treemap_" + str(doc_id) + ".svg"))
    
    
    
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

