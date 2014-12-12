# mkboost 

**mkboot** is an algorithm for learning accurate models for predicting the host class of
viruses based on sequence elements in the viral proteome. mkboost is written in Python2.x,
with computationally intensive segments coded in C inlined in Python.
The algorithm is based on [Adaboost with Alternating Decision Trees]() using
sequence k-mers as features.

This repo contains set of scripts to load the data and run the algorithm, along with a test data set. 
It also contains scripts used for generating the figures in the associated publication. 
This document describes how to download and setup this software package and provides 
instructions on how to run the software on a test dataset of protein sequences of viruses
belonging to the Picornaviridae family.

## Citation

Anil Raj, Michael Dewar, Gustavo Palacios, Raul Rabadan and Chris Wiggins. (2011) *Identifying
Hosts of Families of Viruses: A Machine Learning Approach.* PLoS ONE, 6(12): e27631.

## Dependencies

mkboost depends on 
+ [Numpy](http://www.numpy.org/)
+ [Scipy](http://www.scipy.org/)
+ [Matplotlib](http://www.matplotlib.org/)

A number of python distributions already have these modules packaged in them. It is also
straightforward to install all these dependencies using package managers for MACOSX 
and several Linux distributions.

## Getting the source code

To obtain the source code from github, let us assume you want to clone this repo into a
directory named `proj`:

    mkdir ~/proj
    cd ~/proj
    git clone https://github.com/rajanil/mkboost

To retrieve the latest code updates, you can do the following:

    cd ~/proj/mkboost
    git fetch
    git merge origin/master

Since the software compiles relevant C code inline using [weave](), no further
compilation is necessary.

## Executing the code

The algorithm needs to be run in two steps. 

In the first step, the list of protein sequences
is parsed to generate the set of sequence features, and each virus is represented
in terms of counts of these sequence features. 

    $ python generate_features.py
        + imports: mismatch.py
        + generates the feature space from raw data in <project path>/data/
        + pre-processed data is stored in <project path>/cache/<virus family>_protein/

        1a. mismatch.py
            + functions to generate the set of all k-mers
            and the mismatch feature space.

In the second step, the boosting with ADT
algorithm is run for a fixed maximum number of rounds (maximum model complexity) and
K-fold cross validation is used to determine the test error as a function of
model complexity.

    $ python main.py
        + imports: splitdata.py, boost.py
        + wrapper script to run boosting
        + output files are written to <project path>/cache/<virus family>_protein/

        2a. splitdata.py
            + splits the data into test / train for N-fold CV

        2b. boost.py
            + runs Adaboost

The demo script provided outlines how to run both these steps using the test data.

    $ python demo.py

### Inputs

The inputs that need to be passed to be specified in `demo.py` are
+   name of virus family (assuming the data files are named as shown in `/data`)
+   K (length of k-mers)
+   M (max number of mis-matches allowed)
+   T (number of boosting rounds)
+   model type (trees: full ADTs, stumps: ADTs with depth 1)

### Outputs

The algorithm outputs
+   a text file containing the training and test accuracies at each boosting
    round, along with the predictive k-mer at that round.
+   a pickle file containing the ADT model
+   a pickle file containing the prediction scores at each boosting round
+   a pickle file containing the predicted host label for each virus
    when it is held out as test data.

### Visualization

Scripts to generate the plots in the paper are outlined here.
Figures are output to <project path>/fig/

    3a. plot_boost_auc.py
        - Generates Fig 1 and 2.

    3b. visualize_kmers.py
        - Generates Fig 3.

    3c. visualize_kmers_aligned.py
        - Generates Fig 4.

    3d. visualize_kmers_collapsed.py
        - Generates Fig 5.
