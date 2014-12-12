mkboost
=======

<project path> = mkboost
<virus family> = picorna

Code Summary
    All code for this project is written in Python2.x, with computationally intensive segments coded in C inlined in Python.
    With minor modifications, this code can be run using Python3.x

    Required modules: numpy, scipy, matplotlib

1. generate_features.py
    - imports: mismatch.py
    - generates the feature space from raw data in <project path>/data/
    - pre-processed data is stored in <project path>/cache/<virus family>_protein/

    1a. mismatch.py
        - functions to generate the set of all k-mers
        and the mismatch feature space.

2. main.py
    - imports: splitdata.py, boost.py
    - wrapper script to run boosting
    - output files are written to <project path>/cache/<virus family>_protein/

    2a. splitdata.py
        - splits the data into test / train for N-fold CV

    2b. boost.py
        - runs Adaboost

3. plotting scripts (see http://arxiv.org/abs/1105.5821)
    - figures are output to <project path>/fig/

    3a. plot_boost_auc.py
        - Generates Fig 1 and 2.

    3b. visualize_kmers.py
        - Generates Fig 3.

    3c. visualize_kmers_aligned.py
        - Generates Fig 4.

    3d. visualize_kmers_collapsed.py
        - Generates Fig 5.

Data Summary

    Data files containing protein sequence data and host class for picornaviruses
used in generating results presented in the manuscript (see above) are provided
in <project path>/data/
