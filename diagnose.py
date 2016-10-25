#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: model.py
# Authors: christofs
# Version 0.3.0 (2016-03-20)


import re
import os
import glob
from lxml import etree
import numpy as np
import pandas as pd
import pygal
from pygal.style import Style
from collections import Counter


# Pygal style
mystyle = Style(font_family = "FreeSans",
  title_font_size = 16,
  legend_font_size = 12,
  label_font_size = 9,
  colors = ["#7fcdbb","#41b6c4","#1d91c0","#225ea8","#253494","#081d58", "#071746"])


##########################
# check_topic_indicator
##########################

def read_xml(File): 
    with open(File, "r") as InFile: 
        Data = etree.parse(InFile)
        return Data


def get_indicatordata(DiagnoseData, IndicatorType):  
    Topics = DiagnoseData.findall("topic")
    IndicatorData = []
    for Topic in Topics: 
        Indicator = Topic.get(IndicatorType)
        # coherence|eff_num_words|word-length|corpus_dist
        IndicatorData.append(float(Indicator))
    #print(IndicatorData)
    return IndicatorData


def calculate_statistics(ParamsLabel, IndicatorData): 
    IndicatorMean = np.mean(IndicatorData)
    IndicatorMedian = np.median(IndicatorData)
    IndicatorStd = np.std(IndicatorData)
    IndicatorStats = {"IndicatorMean": IndicatorMean, "IndicatorMedian": IndicatorMedian, "IndicatorStd": IndicatorStd}
    IndicatorStatsSeries = pd.Series(IndicatorStats, name=ParamsLabel)
    return IndicatorStatsSeries


def plot_indicatorstats(AllIndicatorStats, IndicatorType, GraphFileIS): 
    lineplot = pygal.Line(style=mystyle, 
                          x_label_rotation=270)
    lineplot.x_labels = ["100in", "300in", "500in", "1000in", "2000in", "3000in", "6000in"]
    lineplot.add(IndicatorType+"-60tp", AllIndicatorStats.iloc[0:7,0], stroke_style={'width': 4}, dots_size=5)
    lineplot.add(IndicatorType+"-70tp", AllIndicatorStats.iloc[7:14,0], stroke_style={'width': 4}, dots_size=5)
    lineplot.add(IndicatorType+"-80tp", AllIndicatorStats.iloc[14:21,0], stroke_style={'width': 4}, dots_size=5)
    lineplot.add(IndicatorType+"-90tp", AllIndicatorStats.iloc[21:28,0], stroke_style={'width': 4}, dots_size=5)
    lineplot.add(IndicatorType+"-100tp", AllIndicatorStats.iloc[28:35,0], stroke_style={'width': 4}, dots_size=5)
    lineplot.render_to_file(GraphFileIS)



def check_topicindicators(DiagnoseFilePath, IndicatorType, GraphFileIS): 
    """
    Read diagnosis file and extract coherence scores.
    Calculate mean and median coherence schores.
    """
    AllIndicatorStatsList = []    
    for DiagnoseFile in glob.glob(DiagnoseFilePath):
        FileName, Ext = os.path.basename(DiagnoseFile).split(".")
        Prefix, ParamsLabel = FileName.split("_")
        DiagnoseData = read_xml(DiagnoseFile)
        IndicatorData = get_indicatordata(DiagnoseData, IndicatorType)
        IndicatorStatsSeries = calculate_statistics(ParamsLabel, IndicatorData)
        AllIndicatorStatsList.append(IndicatorStatsSeries)
    AllIndicatorStats = pd.concat(AllIndicatorStatsList, axis=1).T.sort_index()
    print(AllIndicatorStats)
    plot_indicatorstats(AllIndicatorStats, IndicatorType, GraphFileIS)


    

#############################
# topicscore_distributions
#############################


def read_topicscores(File): 
    #print("--read_topicscores")
    with open(File, "r") as InFile: 
        Filename, Ext = os.path.basename(File).split(".")
        Prefix, Params = Filename.split("_")
        TopicScoreData = pd.DataFrame.from_csv(InFile, sep="\t")
        TopicScoreData = pd.Series(TopicScoreData.iloc[:,0], name=Params)
        TopicScoreData = TopicScoreData.sort_values(ascending=False)
        print(TopicScoreData)
    return TopicScoreData


def plot_topicscores(AllTopicScoreDistr, GraphFileTSD): 
    print("--plot_topicscores")
    lineplot = pygal.Line(style=mystyle, 
                          x_label_rotation=270,
                          legend_at_bottom=True, 
                          legend_at_bottom_columns=7,
                          stroke_style={'width': 3, 'linecap': 'round'},
                          title = "Topic probability distribution", 
                          x_title = "Topics",
                          y_title = "Topic probability")
    lineplot.x_labels = range(1,80)
    for i in [6,3,5,4,0,2]: 
        lineplot.add(AllTopicScoreDistr[i].name[12:-2], AllTopicScoreDistr[i], show_dots=False)
    lineplot.add("None", AllTopicScoreDistr[1], show_dots=False)
    lineplot.render_to_file(GraphFileTSD)


def topicscore_distributions(TopicScoreFilesPath, GraphFileTSD):
    """
    Read and plot the topic score distributions per topic in the collection.
    """
    AllTopicScoreDistr = []
    for File in glob.glob(TopicScoreFilesPath):
        TopicScoreData = read_topicscores(File)
        AllTopicScoreDistr.append(TopicScoreData)
    plot_topicscores(AllTopicScoreDistr, GraphFileTSD)
    print("Done.")

    
    
    
###############################
# count_words
###############################


def read_file(File): 
    with open(File, "r") as InFile: 
        Text = InFile.read()
        Text = re.sub("\d","", Text)
        Text = re.split("[\W]", Text)
        Text = filter(None, Text)
    return Text

def get_counts(Text): 
    Counts = Counter(Text)
    return Counts

def count_words(TopicsWithWordsFilePath): 
    for File in glob.glob(TopicsWithWordsFilePath):
        Filename, Ext = os.path.basename(File).split(".")
        print(Filename)
        Text = read_file(File)
        Counts = get_counts(Text)
        for word, count in Counts.most_common(5):
            print(word, count)


    
    
"""
def read_wordweights(File): 
    print("--read_wordweights")
    with open(File, "r") as InFile: 
        #print(os.path.basename(File)[-15:-4])
        Data = pd.DataFrame.from_csv(InFile, sep="\t", header=None, index_col=None)
        Data.columns = ["topic", "word", "prob"]
        Data = Data.sort_values(by=["topic", "prob"], ascending=[1, 0])
        WordWeighsData = []
        for Name, Content in Data.groupby(["topic"]):
            OneData = pd.Series(list(Content.loc[:,"prob"].head(50)), name=Name)
            WordWeighsData.append(OneData)
        return WordWeighsData


def plot_wordweights(AllWordWeighsData, GraphFileWWD): 
    print("--plot_wordweights")
    lineplot = pygal.Line(show_legend=False, style=mystyle, range=(0,10000))
    lineplot.x_labels = range(0,50)
    for i in range(0,100): 
        lineplot.add(str(AllWordWeighsData[0][i].name), AllWordWeighsData[0][i], show_dots=True)
        lineplot.render_to_file(GraphFileWWD[:-4]+"0.00001beta.svg")
    lineplot = pygal.Line(show_legend=False, style=mystyle, range=(0,10000))
    lineplot.x_labels = range(0,50)
    for i in range(0,100): 
        lineplot.add(str(AllWordWeighsData[1][i].name), AllWordWeighsData[1][i], show_dots=True)
        lineplot.render_to_file(GraphFileWWD[:-4]+"0.50000beta.svg")
    lineplot = pygal.Line(show_legend=False, style=mystyle, range=(0,10000))
    lineplot.x_labels = range(0,50)
    for i in range(0,100): 
        lineplot.add(str(AllWordWeighsData[2][i].name), AllWordWeighsData[2][i], show_dots=True)
        lineplot.render_to_file(GraphFileWWD[:-4]+"0.10000beta.svg")
    lineplot = pygal.Line(show_legend=False, style=mystyle, range=(0,10000))
    lineplot.x_labels = range(0,50)        
    for i in range(0,100): 
        lineplot.add(str(AllWordWeighsData[3][i].name), AllWordWeighsData[3][i], show_dots=True)
        lineplot.render_to_file(GraphFileWWD[:-4]+"0.00010beta.svg")



   # Read and plot the word weights distributions per topic.
    #AllWordWeightsData = []
    #for File in glob.glob(WordWeightFilesPath):
    #    WordWeighsData = read_wordweights(File)
    #    AllWordWeightsData.append(WordWeighsData)
    #plot_wordweights(AllWordWeightsData, GraphFileWWD)
"""