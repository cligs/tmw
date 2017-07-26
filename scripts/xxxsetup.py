#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup

setup(name="tmw",
    version="0.2.1",
    description="Topic Modeling Workflow",
    long_description="A set of scripts supporting a workflow that includes preparing text files, doing topic modeling with Mallet, postprocessing Mallet output, and visualizing results.",
    url="http://github.com/cligs/tmw",
    author="Christof Schöch",
    author_email="c.schoech@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Operating System :: POSIX :: Linux",
        "License :: MIT",
        "Programming Language :: Python :: 3.4",
    ],
    keywords="topic modeling text processing",
    packages=["tmw"],
    install_requires=[
    "pandas>==0.17.0",
    "numpy>==1.9.0",
    "sklearn",
    "seaborn",
    "lxml",
    "matplotlib",
    "wordcloud",
    ],
    package_data={
        "tmw": ["/extras/en_stopwords.txt"],     
        "tmw": ["/extras/AlegreyaSans-Regular.otf"],     
    },
    zip_safe=False)
