#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: activate_toolbox.py
# Author: Christof Schöch

"""
# Script to add the "tmw" module to your syspath. 
# Run once with appropriate path to use "tmw" with "import".
"""

import sys
import os

# Enter the path to the folder in which your tmw folder is located.
sys.path.append(os.path.abspath("/home/christof/Repos/cligs/tmw/"))

# Optional: Activate to remove a (mistaken or redundant path)    
#sys.path.remove(os.path.abspath("/home/christof/Repos/cligs/tmw"))

# This is for checking whether the path settings are correct.
print(sys.path)
