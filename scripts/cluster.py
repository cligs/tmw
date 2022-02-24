#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: topicpca.py
# Author: #cf


"""
Set of functions to perform clustering on data. 
Built for topic probabilities or word frequencies as input. 
Performs Principal Component Analysis or distance-based clustering.
"""

import os, glob, re
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import pygal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering as AC
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from sklearn import metrics
from scipy.cluster.hierarchy import fcluster


##################################
# Shared functions
##################################

def get_mastermatrix(MastermatrixFile):
    with open(MastermatrixFile, "r") as InFile:
        Mastermatrix = pd.read_csv(InFile)
        #print(Mastermatrix.head())
        return Mastermatrix


def get_topicdata(Mastermatrix): 
    Grouped = Mastermatrix.groupby(by=["idno"])
    TopicData = Grouped.agg(np.mean)
    TopicData = TopicData.iloc[:,5:-1]
    Identifiers = TopicData.index.values
    #print(TopicData)
    #print(Identifiers)
    return TopicData, Identifiers


def get_wordfreqs(WordfreqsFile):
    with open(WordfreqsFile, "r") as InFile:
        Wordfreqs = pd.read_csv(InFile, sep=";")
        #print(Wordfreqs.head())
        return Wordfreqs


def get_freqdata(Wordfreqs, MFW): 
    FreqData = Wordfreqs.iloc[0:MFW,1:]
    FreqDataMean = np.mean(FreqData, axis=1)
    FreqDataStd = np.std(FreqData, axis=1)
    FreqData = FreqData.subtract(FreqDataMean, axis=0)
    FreqData = FreqData.divide(FreqDataStd, axis=0)
    FreqData = FreqData.T
    #print(FreqData.head())
    Identifiers = list(FreqData.index.values)
    #print(Identifiers)
    return FreqData, Identifiers



##################################
# Cluster Analysis with topics
##################################

tc_style = pygal.style.Style(
    background='white',
    plot_background='white',
    font_family = "FreeSans",
    title_font_size = 20,
    legend_font_size = 16,
    label_font_size = 12,
    colors=["#1d91c0","#225ea8","#253494","#081d58", "#071746"])


def get_labels_tc(Identifiers, MetadataFile): 
    with open(MetadataFile, "r") as InFile: 
        Metadata = pd.read_csv(InFile, sep=",")
        Metadata.set_index("idno", inplace=True)
        #print(Metadata.head())
        #print(Identifiers)
        Labels = []
        Colors = []
        GroundTruth = []
        for Item in Identifiers:
            Labels.append(Item)
            Colors.append("darkred")
            GroundTruth.append(0)
            """
            if Metadata.loc[Item,"tc_subgenre"] == "Comédie": 
                Labels.append(Item+"-CO")
                Colors.append("darkred")
                GroundTruth.append(0)
            if Metadata.loc[Item,"tc_subgenre"] == "Tragi-comédie":
                Labels.append(Item+"-TC")
                Colors.append("darkgreen")
                GroundTruth.append(1)
            elif Metadata.loc[Item,"tc_subgenre"] == "Tragédie": 
                Labels.append(Item+"-TR")
                Colors.append("darkblue")
                GroundTruth.append(2)
            """
        #print(Labels)
        return Labels, Colors, GroundTruth


def clusteranalysis(TopicData, Method, Metric):
    """
    docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    """
    # perform the cluster analysis
    LinkageMatrix = linkage(TopicData, method=Method, metric=Metric)
    #print(LinkageMatrix[0:10])
    return LinkageMatrix

    
def make_dendrogram(LinkageMatrix, GraphFolder, 
                    Method, Metric, CorrCoeff, Labels, Colors,
                    DisplayLevels):
    import matplotlib
    if not os.path.exists(GraphFolder):
        os.makedirs(GraphFolder)
    plt.figure(figsize=(12,24))
    plt.title("Plays clustered by topic probabilities", fontsize=14)
    #plt.ylabel("Parameters: "+Method+" method, "+Metric+" metric. CorrCoeff: "+str(CorrCoeff)+".")
    plt.xlabel("Distance\n(Parameters: "+Method+" / "+Metric+")", fontsize=12)
    matplotlib.rcParams['lines.linewidth'] = 1.2
    dendrogram(
        LinkageMatrix,
        p = DisplayLevels,
        truncate_mode="level",
        color_threshold = 1.3,
        show_leaf_counts = True,
        no_labels = False,
        orientation="left",
        labels = Labels, 
        leaf_rotation = 0,  # rotates the x axis labels
        leaf_font_size = 4,  # font size for the x axis labels
        )
    #plt.show()
    plt.savefig(GraphFolder+"dendrogram_"+Method+"-"+Metric+"-"+str(DisplayLevels)+".png", dpi=300, figsize=(12,18), bbox_inches="tight")
    plt.close()


def evaluate_cluster(TopicData, LinkageMatrix, GroundTruth): 
    # check the correlation coefficient
    CorrCoeff, coph_dists = cophenet(LinkageMatrix, pdist(TopicData))
    ## check several cluster evaluation metrics
    Threshold = 2
    FlatClusterNumbers = fcluster(LinkageMatrix, Threshold)
    #print(GroundTruth)
    #print(FlatClusterNumbers)
    ARI = metrics.adjusted_rand_score(GroundTruth, FlatClusterNumbers)
    Homog = metrics.homogeneity_score(GroundTruth, FlatClusterNumbers)
    Compl = metrics.completeness_score(GroundTruth, FlatClusterNumbers) 
    VMeasure = metrics.v_measure_score(GroundTruth, FlatClusterNumbers) 
    print("Evaluation metrics with threshold "+str(Threshold))
    print("CorrCoeff:", CorrCoeff)
    print("adjustedRI:", ARI)
    print("Homogeneity:", Homog)
    print("Completeness:", Compl)
    print("V-Measure:", VMeasure)
    return CorrCoeff, ARI, Homog, Compl, VMeasure


def topiccluster(MastermatrixFile,
                 MetadataFile, 
                 Method,
                 Metric,
                 GraphFolder,
                 DisplayLevels):
    print("Launched topiccluster.")
    Mastermatrix =  get_mastermatrix(MastermatrixFile)
    TopicData, Identifiers = get_topicdata(Mastermatrix)
    Labels, Colors, GroundTruth = get_labels_tc(Identifiers, MetadataFile)
    LinkageMatrix = clusteranalysis(TopicData, Method, Metric)
    CorrCoeff, ARI, Homog, Compl, VMeasure = evaluate_cluster(TopicData, 
                                                              LinkageMatrix, 
                                                              GroundTruth)
    make_dendrogram(LinkageMatrix, GraphFolder, 
                    Method, Metric, 
                    CorrCoeff, Labels, Colors, 
                    DisplayLevels)
    
    print("Done.")











##################################
# Cluster Analysis with words
##################################

tc_style = pygal.style.Style(
    background='white',
    plot_background='white',
    font_family = "FreeSans",
    title_font_size = 20,
    legend_font_size = 16,
    label_font_size = 12,
    colors=["#1d91c0","#225ea8","#253494","#081d58", "#071746"])


def get_labels_wc(Identifiers, MetadataFile): 
    with open(MetadataFile, "r") as InFile: 
        Metadata = pd.read_csv(InFile, sep=";")
        Metadata.set_index("idno", inplace=True)
        #print(Metadata.head())
        #print(Identifiers)
        Labels = []
        Colors = []
        GroundTruth = []
        for Item in Identifiers:
            if Metadata.loc[Item,"tc_subgenre"] == "Comédie": 
                Labels.append(Item+"-CO")
                Colors.append("darkred")
                GroundTruth.append(0)
            if Metadata.loc[Item,"tc_subgenre"] == "Tragi-comédie":
                Labels.append(Item+"-TC")
                Colors.append("darkgreen")
                GroundTruth.append(1)
            elif Metadata.loc[Item,"tc_subgenre"] == "Tragédie": 
                Labels.append(Item+"-TR")
                Colors.append("darkblue")
                GroundTruth.append(2)
        #print(Labels)
        return Labels, Colors, GroundTruth


def clusteranalysis_w(TopicData, Method, Metric):
    """
    docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    """
    # perform the cluster analysis
    LinkageMatrix = linkage(TopicData, method=Method, metric=Metric)
    #print(LinkageMatrix[0:10])
    return LinkageMatrix

    
def make_dendrogram_w(LinkageMatrix, GraphFolder, 
                    Method, Metric, CorrCoeff, Labels, Colors,
                    DisplayLevels):
    import matplotlib
    if not os.path.exists(GraphFolder):
        os.makedirs(GraphFolder)
    plt.figure(figsize=(12,24))
    plt.title("Plays clustered by topic probabilities", fontsize=14)
    #plt.ylabel("Parameters: "+Method+" method, "+Metric+" metric. CorrCoeff: "+str(CorrCoeff)+".")
    plt.xlabel("Distance\n(Parameters: "+Method+" / "+Metric+")", fontsize=12)
    matplotlib.rcParams['lines.linewidth'] = 1.2
    dendrogram(
        LinkageMatrix,
        p = DisplayLevels,
        truncate_mode="level",
        color_threshold = 30,
        show_leaf_counts = True,
        no_labels = False,
        orientation="left",
        labels = Labels, 
        leaf_rotation = 0,  # rotates the x axis labels
        leaf_font_size = 4,  # font size for the x axis labels
        )
    #plt.show()
    plt.savefig(GraphFolder+"dendrogram_"+Method+"-"+Metric+"-"+str(DisplayLevels)+".png", dpi=300, figsize=(12,18), bbox_inches="tight")
    plt.close()


def evaluate_cluster_w(TopicData, LinkageMatrix, GroundTruth): 
    # check the correlation coefficient
    CorrCoeff, coph_dists = cophenet(LinkageMatrix, pdist(TopicData))
    ## check several cluster evaluation metrics
    Threshold = 2
    FlatClusterNumbers = fcluster(LinkageMatrix, Threshold)
    #print(GroundTruth)
    #print(FlatClusterNumbers)
    ARI = metrics.adjusted_rand_score(GroundTruth, FlatClusterNumbers)
    Homog = metrics.homogeneity_score(GroundTruth, FlatClusterNumbers)
    Compl = metrics.completeness_score(GroundTruth, FlatClusterNumbers) 
    VMeasure = metrics.v_measure_score(GroundTruth, FlatClusterNumbers) 
    print("Evaluation metrics with threshold "+str(Threshold))
    print("CorrCoeff:", CorrCoeff)
    print("adjustedRI:", ARI)
    print("Homogeneity:", Homog)
    print("Completeness:", Compl)
    print("V-Measure:", VMeasure)
    return CorrCoeff, ARI, Homog, Compl, VMeasure


def wordcluster(WordfreqsFile,
                AllMFW,
                MetadataFile, 
                Method,
                Metric,
                GraphFolder,
                DisplayLevels):
    print("Launched wordcluster.")
    Wordfreqs =  get_wordfreqs(WordfreqsFile)
    for MFW in AllMFW: 
        FreqData, Identifiers = get_freqdata(Wordfreqs, MFW)
        Labels, Colors, GroundTruth = get_labels_tc(Identifiers, MetadataFile)
        LinkageMatrix = clusteranalysis(FreqData, Method, Metric)
        CorrCoeff, ARI, Homog, Compl, VMeasure = evaluate_cluster(FreqData, 
                                                                  LinkageMatrix, 
                                                                  GroundTruth)
        make_dendrogram_w(LinkageMatrix, GraphFolder, 
                        Method, Metric, 
                        CorrCoeff, Labels, Colors, 
                        DisplayLevels)
    
    print("Done.")










##################################
# PCA with topics
##################################

tp_style = pygal.style.Style(
    background='white',
    plot_background='white',
    font_family = "FreeSans",
    title_font_size = 20,
    legend_font_size = 16,
    label_font_size = 12,
    colors=["#1d91c0","#225ea8","#253494","#081d58", "#071746"])




def get_colors_t(Identifiers, MetadataFile): 
    with open(MetadataFile, "r") as InFile: 
        Metadata = pd.read_csv(InFile, sep=";")
        Labels = list(Metadata.loc[:,"tc_subgenre"])
        Colors = []
        for item in Labels: 
            if item == "Comédie": 
                Colors.append("darkred")
            if item == "Tragi-comédie":
                Colors.append("darkgreen")
            elif item == "Tragédie": 
                Colors.append("navy")
        #print(Colors)
        return Colors
    

def apply_pca_t(TopicData): 
    pca = PCA(n_components=60)
    pca.fit(TopicData)
    Variance = pca.explained_variance_ratio_
    Transformed = pca.transform(TopicData)
    #print(Transformed)
    CumVar = 0
    DimCount = 0
    for item in Variance: 
        CumVar = item+CumVar
        DimCount +=1
        #print("{:02d}".format(DimCount), "{:3f}".format(CumVar))
    return Transformed, Variance
   
   
def make_2dscatterplot_t(Transformed, GraphFolder): 
    if not os.path.exists(GraphFolder):
        os.makedirs(GraphFolder)
    plot = pygal.XY(style=tp_style,
                    stroke=False)
    Data = []
    for i in range(0,391): 
        Point = (Transformed[i][0], Transformed[i][1])
        Data.append(Point)
    plot.add("test", Data)
    plot.render_to_file(GraphFolder+"2dscatter.svg")


def make_3dscatterplot_t(Transformed, Variance, Colors, GraphFolder):
    if not os.path.exists(GraphFolder):
        os.makedirs(GraphFolder)
    azims = range(200,300,10)
    elevs = range(200,300,10)
    for azim in azims: 
        for elev in elevs: 
               
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            allx = []
            ally = []
            allz = []
            for i in range(0,391): 
                allx.append(Transformed[i][0])
                ally.append(Transformed[i][1])
                allz.append(Transformed[i][2])
            ax.scatter(allx, ally, allz, c=Colors, marker="o", s=10, linewidth=0.3)
            plt.setp(ax.get_xticklabels(), fontsize=5)
            plt.setp(ax.get_yticklabels(), fontsize=5)
            plt.setp(ax.get_zticklabels(), fontsize=5)
            ax.set_xlabel("PC1 "+"{:03.2f}".format(Variance[0]), fontsize=8)
            ax.set_ylabel("PC2 "+"{:03.2f}".format(Variance[1]), fontsize=8)
            ax.set_zlabel("PC3 "+"{:03.2f}".format(Variance[2]), fontsize=8)
        
            ax.azim = azim
            ax.elev = elev
            
            #plt.show()
            fig.savefig(GraphFolder+"3dscatter_"+"{:03d}".format(azim)+"-"+"{:03d}".format(elev)+".png", dpi=600, figsize=(3,3), bbox_inches="tight", facecolor="white", transparent=True)
            plt.close()
    

### Main function ###
def topicpca(MastermatrixFile,
             MetadataFile,
             GraphFolder): 
    """
    Coordinating function.
    """
    print("Launched topicpca.")
    Mastermatrix = get_mastermatrix(MastermatrixFile)
    TopicData, Identifiers = get_topicdata(Mastermatrix)
    Colors = get_colors_t(Identifiers, MetadataFile)
    Transformed, Variance = apply_pca_t(TopicData)
    #make_2dscatterplot_t(Transformed, GraphFolder)
    make_3dscatterplot_t(Transformed, Variance, Colors, GraphFolder)
    print("Done.")
    
    
 
 
 
 
 
 
 
 
 
 
################################
# PCA with word frequencies
################################

wd_style = pygal.style.Style(
    background='white',
    plot_background='white',
    font_family = "FreeSans",
    title_font_size = 20,
    legend_font_size = 16,
    label_font_size = 12,
    colors=["#1d91c0","#225ea8","#253494","#081d58", "#071746"])




def get_colors_w(Identifiers, MetadataFile): 
    with open(MetadataFile, "r") as InFile: 
        Metadata = pd.read_csv(InFile, sep=";")
        Metadata.set_index("idno", inplace=True)
        #print(Metadata.head())
        Colors = []
        for Item in Identifiers:
            Label = Metadata.loc[Item,"tc_subgenre"]
            #print(Item, Label)
            if Label == "Comédie": 
                Colors.append("darkred")
            if Label == "Tragi-comédie":
                Colors.append("darkgreen")
            elif Label == "Tragédie": 
                Colors.append("navy")
        #print(Colors)
        return Colors
   
   

def apply_pca_w(FreqData): 
    pca = PCA(n_components=3)
    pca.fit(FreqData)
    Variance = pca.explained_variance_ratio_
    Transformed = pca.transform(FreqData)
    #print(Transformed)
    #print(Variance)
    return Transformed, Variance
   
   
   
def make_2dscatterplot_w(Transformed, GraphFolder): 
    if not os.path.exists(GraphFolder):
        os.makedirs(GraphFolder)
    plot = pygal.XY(style=wd_style,
                    stroke=False)
    Data = []
    for i in range(0,391): 
        Point = (Transformed[i][0], Transformed[i][1])
        Data.append(Point)
    plot.add("test", Data)
    plot.render_to_file(GraphFolder+"2dscatter.svg")



def make_3dscatterplot_w(Transformed, Variance, Colors, GraphFolder, MFW):
    if not os.path.exists(GraphFolder):
        os.makedirs(GraphFolder)
    azims = range(0,350,10)
    elevs = range(180,350,10)
    for azim in azims: 
        for elev in elevs: 
               
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            allx = []
            ally = []
            allz = []
            for i in range(0,391): 
                allx.append(Transformed[i][0])
                ally.append(Transformed[i][1])
                allz.append(Transformed[i][2])
            ax.scatter(allx, ally, allz, c=Colors, marker="o", s=6, linewidth=0.3)
            plt.setp(ax.get_xticklabels(), fontsize=3)
            plt.setp(ax.get_yticklabels(), fontsize=3)
            plt.setp(ax.get_zticklabels(), fontsize=3)
            ax.set_xlabel("PC1 "+"{:03.2f}".format(Variance[0]), fontsize=4)
            ax.set_ylabel("PC2 "+"{:03.2f}".format(Variance[1]), fontsize=4)
            ax.set_zlabel("PC3 "+"{:03.2f}".format(Variance[2]), fontsize=4)
        
            ax.azim = azim
            ax.elev = elev
            
            #plt.show()
            fig.savefig(GraphFolder+"3dscatter_"+"{:04d}".format(MFW)+"mfw-"+"{:03d}".format(azim)+"-"+"{:03d}".format(elev)+".png", dpi=600, figsize=(3,3), bbox_inches="tight")
            plt.close()
    


### Main function ###

def wordpca(WordfreqsFile,
            MetadataFile,
            GraphFolder,
            AllMFW): 
    """
    Coordinating function.
    """
    print("Launched wordpca.")
    WordFreqs = get_wordfreqs(WordfreqsFile)
    for MFW in AllMFW: 
        FreqData, Identifiers = get_freqdata(WordFreqs, MFW)
        Colors = get_colors_w(Identifiers, MetadataFile)
        Transformed, Variance = apply_pca_w(FreqData)
        #make_2dscatterplot_w(Transformed, GraphFolder)
        make_3dscatterplot_w(Transformed, Variance, Colors, GraphFolder, MFW)
    print("Done.")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
