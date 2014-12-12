import numpy as np
import itertools
import timeit
import time
import pdb

def form_all_kmers(A,k):
    """
    Given an alphabet and `k`, this forms an array
    of all possible k-mers using that alphabet.

    Arguments
        A : list
            alphabet - all possible characters
        k : int
            the length of subsequences you're after
    
    Returns
        beta : array
            all possible kmers that can be formed by the alphabet A

    """

    all_kmers = itertools.product(A,repeat=k)
    return np.array([beta for beta in all_kmers])

def form_all_kmers_in_string(k,x):
    """
    Given a string and `k`, this forms all k-mers
    that occur in that string.

    Arguments
        k : int 
            the length of the subseqeunces you're after
        x : string
            the string from which you'd like to form all kmers

    Older code
        >>> strings = np.empty((k, len(x)-k), dtype=str)
        >>> x = list(x)
        >>> for i in range(k):
        >>>     strings[i,:] = x[i:-(k-i)]
        >>> # this is all the kmers
        >>> return np.unique([''.join(kmer) for kmer in strings.T if '*' not in kmer])

    .. note::
        Code implemented is much faster than older code, 
        particularly since it uses list comprehensions.

    """
    kmers = np.unique([x[i:i+k] for i in xrange(len(x)-k) if '*' not in x[i:i+k]])
    return kmers

def gen_features(x,m,beta):
    """
    a feature of `x` is the count in `x` of each kmer in `beta`, where the 
    kmers in `x` are allowed to mismatch each element of beta by `m` 
    mismatches.
    
    Arguments
        x : list
            protein sequence
        m : int
            number of allowed mismatches
        beta : array
            all possible kmers
            
    Returns
        features : array
            count in `x` of each kmer in `beta` varying by `m` mismatches.

    """
    k = len(beta[0])
    y = np.array([list(yi) for yi in form_all_kmers_in_string(k, x)])
    b = np.array([list(bi) for bi in beta])  
    B = len(beta)    

    print "beta contains %s kmers"%B
    print "the current string contains %s kmers"%len(y)

    starttime = time.time()
    count = np.zeros((len(beta),m),dtype=np.int16)
    ms = np.arange(m).reshape(1,m)
    for yi in y:
        count += ((yi!=b).sum(1).reshape(B,1)<=ms)
    print "Feature generation time = %.4f" % (time.time() - starttime)
    
    return count
    
