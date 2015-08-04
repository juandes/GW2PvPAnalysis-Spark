# Guild Wars 2 PvP Analysis - machine learning on Spark

## Overview
***
As a follow up of the report I did previously, titled PvP in _Guild Wars 2: A data analysis_, I decided to run the same machine learning algorithms used in said report, in another framework, Apache Spark, to see how they differ from each other. The original purpose why this was done on the previous report, was to verify if it is possible to predict the outcome of a match using the composition of the team.

The tests were done on Apache Spark 1.4.0, using the Python API.

## Project
***

This repository contains the Python script used in the analysis, the dataset (in .txt) and a codebook explaining the values of the dataset. Regarding the Python script, the _SparkConf_ parameters, are the ones I found suitable for the analysis, however feel free to change them (in particular the master URL, it has to be the URL of your cluster).
