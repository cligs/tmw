#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: classify_bytopics.py
# Author: #cf (2016)

import re
import os
import glob
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from sklearn import cross_validation
from sklearn.linear_model import SGDClassifier
import pygal



################################
# Functions
################################


def define_classifier(ClassifierType): 
    """
    Select and define the type of classifier. 
    Called by "classify_data"
    SVM: http://scikit-learn.org/stable/modules/svm.html
    TRE: http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    """
    if ClassifierType == "SVM": 
        Classifier = svm.SVC(kernel="linear") # linear|poly|rbf
    if ClassifierType == "KNN": 
        Classifier = neighbors.KNeighborsClassifier(n_neighbors=5, weights="distance")
    if ClassifierType == "TRE": 
        Classifier = tree.DecisionTreeClassifier()
    if ClassifierType == "SGD": 
        Classifier = SGDClassifier(loss="log", penalty="l2", shuffle=True)
    return Classifier
    


def get_metadata(MetadataFile, Target): 
    """
    Get a list of labels for the target category
    """
    with open(MetadataFile, "r") as InFile: 
        RawMetadata = pd.DataFrame.from_csv(InFile, sep=";")
        Labels = RawMetadata.loc[:,Target]
        Idnos = RawMetadata.loc[:,"idno"]
        Metadata = pd.concat([Idnos, Labels], axis=1)
        #print(Metadata.head())        
        return Metadata



def get_wordfreqs(WordFreqsFile): 
    """
    Read the document term matrix (CSV file) into a DataFrame.
    """
    with open(WordFreqsFile, "r") as InFile: 
        WordFreqs = pd.read_csv(InFile, sep=";")
        WordFreqs = WordFreqs.rename(columns = {"Unnamed: 0" : "idno"})
        WordFreqs = WordFreqs.set_index("idno", drop=True)
        WordFreqs = WordFreqs.rename(columns = {"idno" : "words"})
        #print(WordFreqs.head())
        return WordFreqs



def make_wordmatrix(WordFreqs, Metadata, Target): 
    """
    Merge the word frequencies with the target metadata.
    As a result, the word frquency table also has a column with the target category.
    """
    #Replace relative frequencies with zscores
    WordMeans = np.mean(WordFreqs, axis=1)
    WordStds = np.std(WordFreqs, axis=1)
    WordFreqs = WordFreqs.subtract(WordMeans, axis=0)        
    WordFreqs = WordFreqs.divide(WordStds, axis=0)
    #print(WordFreqs.head())
    # Reorganize the DataFrame
    WordFreqs = WordFreqs.T
    WordFreqs.reset_index(inplace=True)  
    WordFreqs = WordFreqs.rename(columns = {"index" : "idno"})
    WordFreqs = WordFreqs.iloc[:,0:5001]
    WordMatrix = pd.merge(WordFreqs, Metadata, on="idno")
    #print(WordMatrix.head())
    return WordMatrix



def classify_withwords(WordMatrix, MFW, Target, ClassifierTypes): 
    """
    Classify items using SVM and evaluate accuracy.
    """
    Data = WordMatrix.iloc[:,1:MFW]
    Labels = list(WordMatrix.loc[:,Target])
    #print(list(set(Labels)))
    Results = [MFW]
    Index = ["mfw"]
    for ClassifierType in ClassifierTypes: 
        Classifier = define_classifier(ClassifierType)
        Accuracy = cross_validation.cross_val_score(Classifier, Data, Labels, cv=10, scoring="accuracy")
        #print("Mean accuracy with "+str(ClassifierType)+": %0.3f (+/- %0.3f)" % (Accuracy.mean(), Accuracy.std() * 2))       
        Results.append("{:01.3f}".format(Accuracy.mean()))
        Index.append(str(ClassifierType))
    ResultsName = str(MFW)
    Results = pd.Series(Results, index=Index, name=ResultsName)
    #print(Results)
    return Results


def get_topicscores(TopicScoresFile, NumTopics): 
    """
    Read a CSV file into a DataFrame.
    Make sure each document has its separate idno and segment number.
    """
    with open(TopicScoresFile, "r") as InFile: 
        Columns = ["idno", "seg"] + list(range(0,int(NumTopics)))
        TopicScores = pd.read_csv(InFile, sep="[\t§]", names=Columns, engine="python")
        TopicScores = TopicScores.replace("file.*?/(tc\d\d\d\d)", "\\1", regex=True)
        TopicScores = TopicScores.replace("\.txt", "", regex=True)
        #print(TopicScores.head()) 
        return TopicScores



def make_topicmatrix(TopicScores, Metadata, Target): 
    """
    Merge the topic scores with the target metadata.
    As a result, the topic score table also has a column with the target category.
    Also, calculate topic score means for each complete play from segments.
    """
    # Aggregate topic scores to play level
    TopicScores = TopicScores.groupby("idno")
    #TopicScores = TopicScores.agg(np.mean)
    TopicScores = TopicScores.agg(np.median)
    #print(TopicScores.head())
    # Topic score normalization
    TopicMeans = np.mean(TopicScores, axis=1)
    TopicMedians = np.median(TopicScores, axis=1)
    #TopicStds = np.std(TopicScores, axis=1)
    TopicScores = TopicScores.subtract(TopicMeans, axis=0)        
    #TopicScores = TopicScores.subtract(TopicMedians, axis=0)        
    #TopicScores = TopicScores.divide(TopicStds, axis=0)    
    #print(TopicScores.head())
    TopicScores.reset_index(level=0, inplace=True)
    TopicScores.set_index("idno", drop=True)
    TopicMatrix = pd.merge(TopicScores, Metadata, on="idno")
    #print(TopicMatrix.head())
    return TopicMatrix



def classify_withtopics(TopicMatrix, Target, ClassifierTypes,
                        NumTopics, NumIter, OptInt): 
    """
    Classify items using SVM and evaluate accuracy.
    """
    Data = TopicMatrix.iloc[:,1:int(NumTopics)]
    Labels = list(TopicMatrix.loc[:,Target])
    #print(list(set(Labels)))
    Results = [NumTopics, OptInt]
    Index = ["num-topics", "opt-int"]
    for ClassifierType in ClassifierTypes: 
        Classifier = define_classifier(ClassifierType)
        Accuracy = cross_validation.cross_val_score(Classifier, Data, Labels, cv=10, scoring="accuracy")
        #print("Mean accuracy with "+str(ClassifierType)+": %0.3f (+/- %0.3f)" % (Accuracy.mean(), Accuracy.std() * 2))       
        Results.append("{:01.3f}".format(Accuracy.mean()))
        Index.append(str(ClassifierType))
    ResultsName = str(NumTopics)+"-"+str(OptInt)
    Results = pd.Series(Results, index=Index, name=ResultsName)
    #print(Results)
    return Results

def save_results(AllResults, ResultsFile): 
    #print(AllResults)
    with open(ResultsFile, "w") as OutFile:
        #AllResults.drop(["acc-std", "data-type", "num-iter", "num-words", "target-cat"], axis=1, inplace=True)
        #AllResults.sort_values(by=["num-topics", "opt-int", "classifier"], inplace=True)        
        #AllResults.sort_values(by=["acc-mean"], ascending=False, inplace=True) 
        #AllResults = AllResults.groupby(["num-topics", "classifier"])
        AllResults.to_csv(OutFile, sep="\t", encoding="utf8")
    
    

def classify(TopicScoresFilePath, 
                      WordFreqsFile,
                      MetadataFile, 
                      Target,
                      InputData,
                      ClassifierTypes,
                      WordResultsFile,
                      TopicResultsFile): 
    """
    Classify plays (e.g. into subgenres) based on their topics probabilities and word frequencies.
    Relies on data from tmw (topics-in-texts.csv) and metadata (metadata.csv)
    Finds out how well data can be classified with given data. 
    The idea is to classify data using data built with different parameters
    in order to find the best parameters for a given task. 
    """
    print("Launched.")
    print("Classify by "+Target+".") 
    WordResults = pd.DataFrame()
    TopicResults = pd.DataFrame()
    Metadata = get_metadata(MetadataFile, Target)
    # Get word frequency data and classify with it.
    if "words" in InputData: 
        WordFreqs = get_wordfreqs(WordFreqsFile)
        WordMatrix = make_wordmatrix(WordFreqs, Metadata, Target)
        print("Using word frequency data.")
        for MFW in [100, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]:
            Results = classify_withwords(WordMatrix, MFW, Target, ClassifierTypes)
            WordResults = WordResults.append(Results)
        save_results(WordResults, WordResultsFile)
    ### Get topic probability data and classify with it.
    if "topics" in InputData: 
        print("Using topic probability data.")
        for TopicScoresFile in glob.glob(TopicScoresFilePath):
            Filename, Ext = os.path.basename(TopicScoresFile).split(".")
            NumTopics = re.findall("(\d+)tp", Filename)[0]
            NumIter = re.findall("(\d+)it", Filename)[0]
            OptInt = re.findall("(\d+)in", Filename)[0]
            #print(Filename, NumTopics, NumIter, OptInt)
            TopicScores = get_topicscores(TopicScoresFile, NumTopics)
            TopicMatrix = make_topicmatrix(TopicScores, Metadata, Target)
            Results = classify_withtopics(TopicMatrix, Target, ClassifierTypes,
                                      NumTopics, NumIter, OptInt)    
            TopicResults = TopicResults.append(Results)
        save_results(TopicResults, TopicResultsFile)
    print("Done.")




#########################################
# Plot topic-based classifier results
#########################################

import pygal

my_style = pygal.style.Style(
  background='white',
  plot_background='white',
  font_family = "FreeSans",
  title_font_size = 16,
  legend_font_size = 14,
  label_font_size = 12,
  guide_stroke_dasharray = "0.5,0.5",
  colors=["darkblue","darkcyan","dodgerblue","purple", "#071746"])


def get_topicresults(TopicResultsFile): 
    with open(TopicResultsFile, "r") as InFile: 
        TopicResults = pd.read_csv(InFile, sep=",")
        TopicResults.rename(columns={"Unnamed: 0": "label"}, inplace=True)
        TopicResults.set_index("label", drop=True, inplace=True)
        #print(TopicResults)
        return TopicResults

def get_topicdata(TopicResults, Algorithm):
    #TopicResults.sort_values(Algorithm, ascending=True, inplace=True)
    TopicResults.sort_index(axis=0, ascending=True, inplace=True)
    Data = list(TopicResults.loc[:,Algorithm])
    Labels = TopicResults.index.values
    #print(Labels, Data)
    return Data, Labels


def make_topicplot(TopicResults, GraphFolder): 
    print("make_plot")
    plot = pygal.Line(title = "Classifier performance on topic data" , 
                      x_title = "Input data parameters \n(number of topics, optimize interval)" ,
                      y_title = "Mean accuracy \n(10-fold cross-validation)",
                      legend_at_bottom = True,
                      legend_at_bottom_columns = 5,
                      legend_box_size = 16,
                      style = my_style,
                      x_label_rotation=75,
                      show_x_guides=True,
                      interpolate='cubic')
    for Algorithm in ["SVM", "SGD", "KNN", "TRE"]: 
        Data, Labels = get_topicdata(TopicResults, Algorithm)
        plot.x_labels = Labels
        plot.add(Algorithm, Data, stroke_style={"width": 2,}, dots_size=2)        
    for Algorithm in ["average"]: 
        Data, Labels = get_topicdata(TopicResults, Algorithm)
        plot.add(Algorithm, Data, stroke_style={"width": 2, "dasharray" : "1,1"}, dots_size=2)        
    plot.render_to_file(GraphFolder+"classify-performance_topics.svg")












def plot(WordResultsFile,
         TopicResultsFile,
         GraphFolder): 
    """
    Function to make plots from the classification performance data.
    """
    TopicResults = get_topicresults(TopicResultsFile)
    TopicData = make_topicplot(TopicResults, GraphFolder)


































