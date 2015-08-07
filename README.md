tmw - Topic Modeling Workflow
=============================

## What is tmw?

tmw is a python module for topic modeling, including some preprocessing of texts and some postprocessing of topic model data.

## Requirements

* The bad news: tmw has been developed for and tested only on Linux (Ubuntu 14.04)
* Make sure you have Python 3 (tested with 3.4), Mallet (tested with 2.0.7) and TreeTagger installed.
* Make sure you have the Python 3 packages numpy, pandas, matplotlib, lxml, scipy, seaborn, wordcloud. 

## Usage

* Download the module files tmw.py (the module) and my_tmw.py (the configuration file) to a convenient location
* Make sure all your document files are in one directory (your working directory)
* If you would like to use a stoplist when using Mallet, add this to a subdirectory called "extras"
* The workflow accepts TEI-encoded or plain text documents as input and expects each file to contain one document
* Set the working directory ("wdir") in the configuration file
* (Optionally, you may adapt the module itself in the "tmw.py" file.) 
* Activate or deactivate functions in the configuration file by un-commenting them as needed
* For each function you want to use, set the parameters as necessary.
* Run the workflow by executing the configuration file in a Python IDE.

## Contact

Christof Schöch, c.schoech@gmail.com
See also: http://github.com/cligs
