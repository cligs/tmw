#!/usr/bin/env python3
# Filename: my_tmw.py
# Author: #cf
# Version 0.2.0 (2015-08-27)


import boxplot

### Set the general working directory.
WorkDir = "/media/christof/data/Dropbox/0-Analysen/2016/gddh-dhq/tc391zz/" # end with slash.
NumTopics = 60

### progression
MastermatrixFile = WorkDir + "9_aggregates/060tp-0300in/mastermatrix.csv"
FirstwordsFile = WorkDir + "9_aggregates/060tp-0300in/firstWords.csv"
GraphFolder = WorkDir + "9_visuals/060tp-1000in/progression/"
#boxplot.progression(MastermatrixFile,
#                    FirstwordsFile,
#                    GraphFolder,
#                    NumTopics)


### complex progression (bins and subgenre)
MastermatrixFile = WorkDir + "9_aggregates/060tp-0300in/mastermatrix.csv"
FirstwordsFile = WorkDir + "9_aggregates/060tp-0300in/firstWords.csv"
GraphFolder = WorkDir + "9_visuals/060tp-0300in/complex/"
boxplot.complex(MastermatrixFile,
                FirstwordsFile,
                GraphFolder,
                NumTopics)

                           

### comparison (of subgenres)
MastermatrixFile = WorkDir + "9_aggregates/060tp-0300in/mastermatrix.csv"
FirstwordsFile = WorkDir + "9_aggregates/060tp-0300in/firstWords.csv"
GraphFolder = WorkDir + "9_visuals/060tp-0300in/comparison/"
#boxplot.comparison(MastermatrixFile,
#                    FirstwordsFile,
#                    GraphFolder,
#                    NumTopics)

