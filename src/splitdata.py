import numpy as np
import random as rand
import pdb

def cv_multiclass_fold(Y,num_fold=10):
    """
	split the data indices into test sets with elements
	of each class distributed among test sets
	in a balanced way.

    Arguments
        Y : array
            Data labels where rows correspond to labels
            and columns to samples. This is an array of {1,-1}.

    Kwargs
        num_fold : int
            number of cross-validation folds
            to split the data into.

    Returns
        index_list : list
            list of `num_fold` lists of data indices.

    """
	
    (K,N) = Y.shape
    indices = dict()
    Nk = dict()
    for k in range(K):
        # select indices belonging to class k
        indices[k] = list((Y[k,:]==1).nonzero()[0])
        rand.shuffle(indices[k])
        Nk[k] = len(indices[k])/num_fold
	
    index_list = []

    for k in range(K):
        for i in range(num_fold-1):
            # split class-k indices into num_fold random sets
            try:
                index_list[i].extend(indices[k][Nk[k]*i:Nk[k]*(i+1)])
            except IndexError:
                index_list.append([])
                index_list[i].extend(indices[k][Nk[k]*i:Nk[k]*(i+1)])
        try:
            index_list[num_fold-1].extend(indices[k][Nk[k]*(num_fold-1):])
        except IndexError:
            index_list.append([])
            index_list[num_fold-1].extend(indices[k][Nk[k]*(num_fold-1):])

    return index_list

def cv_split(Xt,Yt,indices):
	"""Given the test set indices, return train+test data

    Arguments
        Xt : float array
            Feature data where rows correspond to features
            and columns to samples.
        Yt : float array
            Label data where rows correspond to labels
            and columns to samples. This is an array of {1,-1}.
        indices : list
            data indices selected to be the test set

    Returns
        X : float array 
            Training data where rows correspond to features
            and columns to samples.
        Y : float array
            Training labels where rows correspond to labels
            and columns to samples. This is an array of {1,-1}.
        x : float array 
            Testing data with similar row, column properties
            as training data.
        y : int array
            Testing labels with similar row, column properties
            as training labels.

    .. warning::
        Transposing an array in Numpy, oddly, changes the order
        of the array from C contiguous to Fortran contiguous.
        Since data from this function will be processed by a C
        code in `boost` module that expects C contiguous arrays, 
        the order of the arrays are explicitly set to be C contiguous. 
        This can be changed once the C code in the `boost` module is
        generalized to allow for any array order. Until then, the 
        order of the array has to explicitly be set to C-contiguous.

	"""

	if Yt.shape[1]==1:
		tndx = np.zeros(Yt.shape,dtype='int')
	else:
		tndx = np.zeros((Yt.shape[1],1),dtype='int')
	tndx[indices,0] = 1
	Tndx = 1 - tndx

	# training data
	X = np.array(Xt[:,Tndx.nonzero()[0]], copy=True, order='C').astype('float')
	if Yt.shape[1]==1:
		Y = Yt[Tndx.nonzero()[0]]
	else:
		Y = np.array(Yt[:,Tndx.nonzero()[0]], copy=True, order='C')

	# testing data
	x = np.array(Xt[:,tndx.nonzero()[0]], copy=True, order='C').astype('float')
	if Yt.shape[1]==1:
		y = Yt[tndx.nonzero()[0]]
	else:
		y = np.array(Yt[:,tndx.nonzero()[0]], copy=True, order='C')

	return X, Y, x, y
