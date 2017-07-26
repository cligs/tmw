#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: progression_boxplot.py
# Authors: #cf
# Version 0.1.0 (2016-10)


"""
Boxplot visualisation of topics over text time, with variation.
"""

import os
import glob
import pandas as pd
import numpy as np
import pygal


##########################
# Shared functions
##########################



def get_firstwords(FirstwordsFile):
    """
    Function to load list of top topic words into dataframe.
    """
    with open(FirstwordsFile, "r") as InFile: 
        Firstwords = pd.read_csv(InFile, header=None)
        Firstwords.drop(0, axis=1, inplace=True)
        Firstwords.rename(columns={1:"topicwords"}, inplace=True)
        Firstwords.index = Firstwords.index.astype(np.int64)        
        #print(Firstwords)
        return(Firstwords)



def get_mastermatrix(MastermatrixFile):
    with open(MastermatrixFile, "r") as InFile:
        Mastermatrix = pd.read_csv(InFile)
        #print(Mastermatrix.head())
        return Mastermatrix


##########################
# progression (simple)
##########################


pg_style = pygal.style.Style(
  background='white',
  plot_background='white',
  font_family = "FreeSans",
  title_font_size = 20,
  legend_font_size = 16,
  label_font_size = 12,
  colors=["#1d91c0","#225ea8","#253494","#081d58", "#071746"])



def get_binneddata(Mastermatrix):
    BinnedData = Mastermatrix.groupby(by=["binID"])
    return BinnedData


def select_data(BinnedData, Topic):
    SelectedData = []
    for name, group in BinnedData:
        group = group.loc[:,str(Topic)]
        SelectedData.append(group)
    return SelectedData


def make_boxplot(SelectedData, Topic, Firstwords, GraphFolder):
    if not os.path.exists(GraphFolder):
        os.makedirs(GraphFolder)
    Boxplot = pygal.Box(legend_at_bottom=True,
                        legend_at_bottom_columns=5,
                        pretty_print=True,
                        style=pg_style,
                        box_mode="tukey",
                        title = str(Firstwords.iloc[Topic,0]))
    for i in range(0,5): 
        Boxplot.add("Segment"+str(i+1), SelectedData[i])
    Boxplot.render_to_file(GraphFolder+"boxplot_"+"{:03d}".format(Topic)+".svg")


def progression(MastermatrixFile,
                FirstwordsFile,
                GraphFolder,
                NumTopics):
    """
    Function to plot topic development over textual progression using boxplots.
    """
    print("Launched progression.")
    Firstwords = get_firstwords(FirstwordsFile)
    Mastermatrix = get_mastermatrix(MastermatrixFile)
    BinnedData = get_binneddata(Mastermatrix)
    for Topic in range(0,NumTopics): 
        SelectedData = select_data(BinnedData, Topic)
        make_boxplot(SelectedData, Topic, Firstwords, GraphFolder)
    print("Done.")
    



##########################
# complex progression
##########################


cx_style = pygal.style.Style(
  background='white',
  plot_background='white',
  font_family = "FreeSans",
  title_font_size = 20,
  legend_font_size = 16,
  label_font_size = 12,
  colors=["#1d91c0", "#1d91c0", "#1d91c0", "#1d91c0", "#1d91c0", "#071746", "#071746", "#071746", "#071746", "#071746"])


def get_complexbinneddata(Mastermatrix):
    BinnedData = Mastermatrix.groupby(by=["tc_subgenre", "binID"])
    #print(BinnedData.head())
    return BinnedData


def select_complexdata(BinnedData, Topic):
    SelectedData = []
    Labels = []
    for label, group in BinnedData:
        if "Comédie" in label or "Tragédie" in label:
            Labels.append(label)
            group = group.loc[:,str(Topic)]
            SelectedData.append(list(group))
    #print(Labels)
    return SelectedData, Labels


def make_complex(SelectedData, Labels, Topic, Firstwords, GraphFolder):
    if not os.path.exists(GraphFolder):
        os.makedirs(GraphFolder)
    XLabels = [str(Label[1]+1) for Label in Labels]
    Boxplot = pygal.Box(show_legend = False,
                        #legend_at_bottom = True,
                        #legend_at_bottom_columns = 15,
                        x_labels = XLabels,
                        x_label_rotation=0,
                        pretty_print=True,
                        style = cx_style,
                        box_mode = "tukey",
                        title = str(Firstwords.iloc[Topic,0]),
                        x_title = "Comédies _________________ Tragédies")
    for i in range(0,len(XLabels)): 
        #print(str(Labels[i]))
        Boxplot.add(str(Labels[i]), SelectedData[i])
    Boxplot.render_to_file(GraphFolder+"boxplot_"+"{:03d}".format(Topic)+".svg")




def complex(MastermatrixFile,
            FirstwordsFile,
            GraphFolder,
            NumTopics):
    """
    Function to plot topic development over textual progression using boxplots.
    """
    print("Launched complex.")
    Firstwords = get_firstwords(FirstwordsFile)
    Mastermatrix = get_mastermatrix(MastermatrixFile)
    BinnedData = get_complexbinneddata(Mastermatrix)
    for Topic in range(0,60): 
        SelectedData, Labels = select_complexdata(BinnedData, Topic)
        make_complex(SelectedData, Labels, Topic, Firstwords, GraphFolder)
    print("Done.")





##########################
# comparison
##########################


cp_style = pygal.style.Style(
  background='white',
  plot_background='white',
  font_family = "FreeSans",
  title_font_size = 20,
  legend_font_size = 16,
  label_font_size = 12,
  colors=["#4078a5", "#7eaacd", "#1d3549"])


def get_comparisonbinneddata(Mastermatrix):
    BinnedData = Mastermatrix.groupby(by=["tc_subgenre"])
    return BinnedData


def select_comparisondata(BinnedData, Topic):
    SelectedData = []
    Labels = []
    for label, group in BinnedData:
        Labels.append(label)
        group = group.loc[:,str(Topic)]
        SelectedData.append(group)
    return SelectedData, Labels


def make_comparisonboxplot(SelectedData, Labels, Topic, Firstwords, GraphFolder):
    if not os.path.exists(GraphFolder):
        os.makedirs(GraphFolder)
    XLabels = [str(Label) for Label in Labels]
    Boxplot = pygal.Box(show_legend=False, 
                        #legend_at_bottom=False,
                        #legend_at_bottom_columns=5,
                        pretty_print=True,
                        style=cp_style,
                        box_mode="tukey", #tukey|1.5IQR|stdev
                        title = str(Firstwords.iloc[Topic,0]),
                        x_labels = XLabels)
    for i in range(0,3): 
        Boxplot.add("Segment"+str(i+1), SelectedData[i])
    Boxplot.render_to_file(GraphFolder+"boxplot_"+"{:03d}".format(Topic)+".svg")


def comparison(MastermatrixFile,
                FirstwordsFile,
                GraphFolder,
                NumTopics):
    """
    Function to plot topic development over textual progression using boxplots.
    """
    print("Launched comparison.")
    Firstwords = get_firstwords(FirstwordsFile)
    Mastermatrix = get_mastermatrix(MastermatrixFile)
    BinnedData = get_comparisonbinneddata(Mastermatrix)
    for Topic in range(0,NumTopics): 
        SelectedData, Labels = select_comparisondata(BinnedData, Topic)
        make_comparisonboxplot(SelectedData, Labels, Topic, Firstwords, GraphFolder)
    print("Done.")




