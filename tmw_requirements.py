#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: requirements.py
# Author: Christof Schöch

"""
# Script to print versions of all dependencies. 
"""




# TESTING PYTHON PACKAGE VERSION REQUIREMENTS

import numpy
print("       numpy: current = ", numpy.__version__, " | required =  1.9.3")
import pandas
print("      pandas: current =", pandas.__version__, " | required = 0.16.1")
import seaborn
print("     seaborn: current = ", seaborn.__version__, " | required =  0.5.1")
import matplotlib
print("  matplotlib: current = ", matplotlib.__version__, " | required =  1.4.3")
import nltk
print("        nltk: current = ", nltk.__version__, " | required =  3.0.4")
#import PIL
#print("         PIL: current = ", PIL.__version__, " | required =  3.0.4")
#import subprocess
#print("  subprocess: current version: ", subprocess.__version__, " | required: 1.4.3")
#import lxml
#print("  lxml: current version: ", lxml.__version__, " | required: 1.4.3")
#import wordcloud
#print("  wordcloud: current version: ", wordcloud.__version__, " | required: 1.4.3")



# PYTHON PACKAGE REQUIREMENTS WITHOUT VERSION INFORMATION 

# Recent subprocess
# Recent lxml
# Recent wordcloud
# Recent PIL



# FOR INFORMATION: OTHER REQUIREMENTS OF TMW

# TreeTagger including language parameter files.
# Mallet 2.0.7 or >2.0.8
# AlegreyaSansRegular.otf 