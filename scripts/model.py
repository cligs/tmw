#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: model.py
# Authors: christofs
# Version 0.3.0 (2016-03-20)

##################################################################
###  Topic Modeling Workflow (tmw)                             ###
##################################################################

##################################################################
###  model.py - actual topic modeling with Mallet              ###
##################################################################


import os
import subprocess
from os.path import join

#################################
# call_mallet_import            #
#################################


def call_mallet_import(MalletPath, 
						TextFolder, 
						MalletFolder, 
						CorpusFile, 
						StoplistProject):
	"""
	Function to import text data into Mallet.
	"""
	print("\nLaunched call_mallet_import.")    
	if not os.path.exists(MalletFolder):
		os.makedirs(MalletFolder)    
	### Fixed parameters.
	TokenRegex = "'\p{L}[\p{L}\p{P}]*\p{L}'"
	### Building the command line command    
	command = MalletPath + " import-dir --input " + TextFolder + " --output " + CorpusFile + " --keep-sequence --token-regex " + TokenRegex + " --remove-stopwords TRUE --stoplist-file " + StoplistProject
	## Make the call
	print(command)
	subprocess.call(command, shell=True)
	print("Done.\n")



#################################
# call_mallet_modeling          #
#################################

def call_mallet_modeling(MalletPath, 
						CorpusFile, 
						ModelFolder, 
						NumTopics, 
						NumIterations,
						OptimizeIntervals,
						NumRepetitions,
						NumTopWords,
						NumThreads):
	"""Function to perform topic modeling with Mallet."""
	print("\nLaunched call_mallet_modeling.")

	### Getting ready.
	if not os.path.exists(ModelFolder):
		os.makedirs(ModelFolder)

	### Constructing Mallet command from parameters.
	for Repetition in range(NumRepetitions):
		#Repetition = 4
		print("modeling repetition number " + str(Repetition) + "...")
		for Topics in NumTopics: 
			for Iterations in NumIterations: 
				for Interval in OptimizeIntervals:
					Params = str(Topics)+"tp-"+str(Iterations)+"it-"+str(Interval)+"in-"+str(Repetition)
					print("Now modeling with:", Params)
					DocTopicsMax = Topics
					### Output parameters                
					word_topics_counts_file = join(ModelFolder, "words-by-topics_" + Params + ".txt")
					topic_word_weights_file = join(ModelFolder, "word-weights_" + Params + ".csv")
					output_topic_keys = join(ModelFolder, "topics-with-words_" + Params + ".csv")
					output_doc_topics = join(ModelFolder, "topics-in-texts_" + Params + ".csv")
					output_topic_state = join(ModelFolder, "topic_state_" + Params + ".gz")
					DiagnosticsFile = join(ModelFolder, "diagnostics_" + Params + ".xml")
					# Building the call
					command = (MalletPath +" train-topics" +
							  " --input "+ CorpusFile +
							  " --num-topics "+ str(Topics) +
							  " --num-iterations " + str(Iterations) +
							  " --num-top-words " + str(NumTopWords) +
							  " --word-topic-counts-file "+ word_topics_counts_file +
							  " --topic-word-weights-file "+ topic_word_weights_file +
							  " --output-topic-keys "+ output_topic_keys +
							  " --output-doc-topics "+ output_doc_topics +
							  " --doc-topics-max "+ str(DocTopicsMax) +
							  " --output-state " + output_topic_state +
							  " --diagnostics-file "+DiagnosticsFile + 
							  " --num-threads " + str(NumThreads))
					if Interval is not None:		  
						command = command + " --optimize-interval "+ str(Interval)
					#print(command)
					subprocess.call(command, shell=True)
	print("Done.\n")


