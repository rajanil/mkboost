import numpy as np
import scipy.weave as weave
import scipy.stats.stats as stats
import cPickle
import os, time
import pdb

# machine epsilon
EPS = np.finfo(np.double).tiny

# sum function that maintains array shape length
sum = lambda x,axes: np.apply_over_axes(np.sum,x,axes)
prod = lambda x,y: x*y

class ADT(dict):
    """A dictionary to store the Alternating Decision Tree model.

    An Alternating Decision Tree contains decision nodes and
    output nodes. Decision nodes are implemented as a `Node`
    class in this module. In the ADT dictionary, keys correspond
    to the boosting round in which the node is added and can be
    used as a numeric identifier for that decision node. Values
    are lists containing an instance of the decision node and
    associated output nodes. Adding a child node reveals only
    inheritance from its immediate parent.

    The ADT is initialized by passing the attributes of the Root
    Node.

    Arguments
        alpha : float
            weight of the root node (bias term in the ADT model).
        v : array
            vote vector of the root node

    """

    def __init__(self, alpha, v):
        self[-1] = [Node(name='Root'), [alpha, v, []]]

    def add_decision_node(self, id, name, threshold):
        """Add a decision node to the ADT.

        Arguments
            id : int
                boosting round in which the node is added.
            name : str
                string identifying the feature associated with the
                decision node.
            threshold : float
                the threshold on the set of values of the feature
                associated with the decision node

        """

        self[id] = [Node(name=name, threshold=threshold), [], []]

    def add_output_node(self, id, alpha, v, output_type=0):
        """Add an output node to the ADT.

        Arguments
            id : int
                boosting round in which the node is added.
            alpha : float
                weight of the binary-valued function associated with 
                the output node
            v : int array
                vote vector of the binary-valued function associated
                with the output node

        Kwargs
            output_type : {0,1}
                this determines which of the two output nodes is
                being added

        .. note::
            There are two output nodes for each decision node, corresponding
            to the two binary-valued functions :math:`\psi` and :math:`\\tilde{\psi}` 
            (see `paper <http://arxiv.org/abs/1105.5821>`_ for details). The 
            parameter `output_type` determines which of these two output 
            nodes is being added to the model.

        """

        self[id][output_type+1] = [alpha, v, []]

    def add_child(self, parent, child, output_type=0):
        """Append the new decision node as a child of an existing parent
        node.

        Arguments
            parent : int
                node identifier of the parent decision node to which new
                decision node is being added.
            child : int
                node identifier of the new decision node.

        Kwargs
            output_type : {0,1}
                this determines to which of the two output nodes of the
                parent decision node is the new decision node being added.

        """

        try:
            self[parent][output_type+1][2].append(child)
        except IndexError:
            print (parent, output_type, self[parent])


class Node:
    """A class to hold attributes of decision nodes in an ADT.

    Kwargs
        name : str
            string identifying the feature associated with the
            decision node.
        threshold : int or float
            the threshold on the set of values of the feature
            associated with the decision node

    """

    def __init__(self, name=None, threshold=None):
        if name:
            self.name = name
        else:
            self.name = 'Root'
        if threshold:
            self.threshold = threshold


def adaboost(X, Y, x, y, T, output_file=None, kmer_dict=None, \
            model='stump', predicted_labels=None, test_indices=None):
    """This function runs Adaboost, given some training and
    testing data.

    Arguments
        X : float array 
            Training data where rows correspond to features
            and columns to samples.
        Y : float array
            Training labels where rows correspond to labels
            and columns to samples. This is an array of {1,-1}.
        x : float array 
            Testing data with similar row,column properties
            as training data.
        y : float array
            Testing labels with similar row,column properties
            as training labels.
        T : int
            Number of boosting rounds

    Kwargs
        output_file : str
            A full file path to which results can be written 
            during code execution. If not provided, the code 
            creates a results directory one level up and writes 
            to it.
        kmer_dict : dict
            A dictionary with row indices of data as keys and 
            some application relevant identifier of the row
            as value. Values should be strings.
        model : {**tree**, stump}
            ADT model type
            * tree = full ADT
            * stump = ADT with depth 1
    
    Returns
        adt : ADT instance
            The final ADT model.
        performance : array
            An array containing train/test accuracy of the model
            at each boosting round, along with runtime for each
            round.

    .. note::
        * D kmers
        * N virus sequences
        * L Host Classes

        `phi[label]` dictionary stores outputs of each binary function
        and the inheritance of each decision node. Its values
        contain a function :math:`\psi` (:math:`1 \\times N` array), 
        a scalar :math:`\\alpha`, and a vote vector `v` (:math:`L \\times 1` 
        array). Their product is a :math:`L \\times N`
        array which represents the contribution of the new
        output node to the total classification of each object.
        f is a rank 3 array (:math:`L \\times N \\times T+1`) that 
        stores the output of the ADT for each virus sequence at each round
        of boosting. 

    Anil Raj, Michael Dewar, Gustavo Palacios, Raul Rabadan, Chris Wiggins.
    Identifying Hosts of Families of Viruses: a Machine Learning Approach
    arXiv:1105.5821v1 [q-bio.QM] 29 May 2011

    """

    (D,N) = X.shape
    L = Y.shape[0]
    n = x.shape[1]
    if test_indices:
        test_indices.sort()

    # create output file, if not provided
    # create a data directory, one level above
    if not output_file:
        cwd = os.getcwd().split(os.sep)[:-1]
        output_file = cwd[:-1].extend(['data','output.txt'])
        output_file = os.sep.join(output_file)
        os.makedirs(os.sep.join(cwd[:-1].extend(['data'])))

    # Initialize data structures to store model and output
    performance = np.zeros((T+1,5),dtype=float)

    # Each example has equal weight of 1/NL
    w = np.ones(Y.shape,dtype=float)/(N*L)

    # phi stores the output of each binary-valued function
    # for train and test data
    phi = {'train': dict(), 'test': dict()}

    # f = output of the ADT at each round, for train & test data
    f = {'train': np.zeros((L,N,T+1), dtype=float),
                'test': np.zeros((L,n,T+1), dtype=float)}
    
    starttime = time.time()

    # v = vote vector
    v = (sum((w*Y),[1])>0)*2.-1.

    # compute cumulative weights
    Wplus = w[Y*v>0].sum()
    Wminus = w[Y*v<0].sum()

    # alpha = coefficient of weak rule
    alpha = 0.5*np.log((Wplus+EPS)/(Wminus+EPS))

    # alpha is kept positive 
    #vote vector captures the sign of the rule coefficient
    if alpha<0:
        alpha = np.abs(alpha)
        v = -1*v

    # update phi dictionary. Array represents \psi
    phi['train'][-1] = [[np.ones((1,N),dtype=float),alpha,v]]
    phi['test'][-1] = [[np.ones((1,n),dtype=float),alpha,v]]

    # initialize ADT
    adt = ADT(alpha,v)

    # compute the prediction of the ADT for all train/test samples

    # train/test (keys), data dictionary (values)
    for label,data in phi.items():
        # data.values() has one output node with a list [\psi, v and \alpha]
        for node in data.values():
            # child(ren) are \psi, v, \alpha
            for child in node:
                # Entries of product represent the contribution of the new weak
                # rule to the classification of each virus.
                f[label][:,:,0] += reduce(prod,child)

    # updated weights
    w = np.exp(-f['train'][:,:,0]*Y)
    w = w/w.sum()

    # compute classification error at round 0
    performance[0,:4] = compute_auc(f['train'][:,:,0],f['test'][:,:,0],Y,y)
    performance[0,4] = time.time() - starttime
    
    # write intermediate output to file
    handle = open(output_file,'a')
    to_write = [-1, 'root', 'None'] 
    to_write.extend(list(performance[0,:]))
    handle.write('\t'.join(map(str,to_write))+'\n')
    handle.close()

    # starting boosting rounds
    for t in range(T):
        starttime = time.time()

        # choose the appropriate (path,feature) for the next binary-valued function
        path, feature, decision, threshold \
            = get_new_function(X, Y, phi['train'], w, model)

        # slices the feature space to an array which represents the kmer feature the
        # new weak rule has picked. Returns a 1xN array of {True, False}
        PX = X[feature:feature+1,:]<threshold
        px = x[feature:feature+1,:]<threshold

        phi['train'][t] = []
        phi['test'][t] = []
        adt.add_decision_node(t, kmer_dict[feature], threshold)
        adt.add_child(path, t, decision)

        # iterates over the two output nodes that a decision node can have.
        # 0 indicates a "yes" output and 1 indicates a "no" output.
        for ans in [0,1]:
            
            # compute output of decision function
            # the train_phi is based on its value prior to decision round t
            train_phi = phi['train'][path][decision][0] * (ans+(-1.)**ans*PX)
            test_phi = phi['test'][path][decision][0] * (ans+(-1.)**ans*px)

            # calculate optimal value of (alpha,v) for the new
            # binary-valued function
            v = (sum(w*Y*train_phi,[1])>0)*2.-1.
            Wplus = w[Y*v*train_phi>0].sum()
            Wminus = w[Y*v*train_phi<0].sum()
            alpha = 0.5*np.log((Wplus+EPS)/(Wminus+EPS))
            # alpha is always kept positive
            if alpha<0:
                alpha = np.abs(alpha)
                v = -1*v

            # Update Tree and prediction dictionary
            phi['train'][t].append([train_phi,alpha,v])
            phi['test'][t].append([test_phi,alpha,v])
            adt.add_output_node(t, alpha, v)

            # compute the prediction of the ADT for all train/test samples

            # train/test (keys), data dictionary (values)
            for label,data in phi.items():
                # data.values() has two output nodes with each with a list [\psi, v and \alpha]
                for node in data.values():
                    # child(ren) are \psi, v, \alpha
                        for child in node:
                            # Entries of product represent the contribution of the new weak
                            # rule to the classification of each virus.
                            f[label][:,:,t+1] += reduce(prod,child)
                
        # updated weights
        w = np.exp(-f['train'][:,:,t+1]*Y)
        w = w/w.sum()        

        # compute the test / train AUC and test / train classification errors
        performance[t+1,:4] = compute_auc(f['train'][:,:,t+1], f['test'][:,:,t+1], Y, y)
        predicted_labels[test_indices,t] = f['test'][:,:,t+1].argmax(0)
        performance[t+1,4] = time.time() - starttime

        # output data
        handle = open(output_file,'a')
        to_write = [t, kmer_dict[feature], threshold]
        to_write.extend(list(performance[t+1,:]))
        handle.write('\t'.join(map(str,to_write))+'\n')
        handle.close()
    
    return adt, f, performance, predicted_labels


def compute_auc(train, test, Y, y):
    """Computes measures of accuracy for train and test data.

    Computes the ROC curve and the area under that curve, as a 
    measure of classification accuracy. The threshold corresponding
    to the point on the ROC curve farthest from `y=x` line is selected
    and fraction of correct predictions corresponding to that
    threshold is returned.

    Arguments
        train : float array
            Array of predictions of the model on training data where rows
            correspond to labels and columns correspond to samples.
        test : float array
            Array of predictions of the model on testing data where rows
            correspond to labels and columns correspond to samples.
        Y : float array
            Training labels where rows correspond to labels
            and columns to samples. This is an array of {1,-1}.
        y : float array
            Testing labels where rows correspond to labels
            and columns to samples. This is an array of {1,-1}.

    Returns
        performance : float array
            Array containing the AUC for training data, classification 
            accuracy for training data, AUC for testing data and
            classification accuracy for testing data, in that order. 

    .. note::
        * For binary-class classification, AUC is proportional to the Mann-Whitney U test statistic which computes a measure   of the separation between values of positive labels and negative labels.
        
        * For multi-class classification, this formula for computing classifier AUC is one of many. A more principled way      would involve computing the Volume under an ROC surface.

    """

    # computing train AUC
    NP = (Y==1).sum()
    NM = (Y==-1).sum()
    try:
        U = stats.mannwhitneyu(train[(Y==1)],train[(Y==-1)])
        train_auc = 1.-U[0]/(NP*NM)
    except ValueError:
        train_auc = 0.5
    
    # computing test AUC
    NP = (y==1).sum()
    NM = (y==-1).sum()
    try:
        U = stats.mannwhitneyu(test[(y==1)],test[(y==-1)])
        test_auc = 1.-U[0]/(NP*NM)
    except ValueError:
        test_auc = 0.5

    # accuracy = number of examples where argmax of prediction
    # equals true label

    # train accuracy 
    train_accuracy = (train.argmax(0)-Y.argmax(0)==0).sum()/float(Y.shape[1])

    # test accuracy     
    test_accuracy = (test.argmax(0)-y.argmax(0)==0).sum()/float(y.shape[1])

    return np.array([train_auc, train_accuracy, test_auc, test_accuracy])


def get_new_function(X, Y, phi, w, model='tree'):
    """This function finds the best feature to add to an ADT.

    This function computes the minimum exponential loss achieved
    by each potential decision node and selects the one that
    has the least exponential loss.

    Arguments
        X : float array
            Data array where rows correspond to features and columns
            correspond to samples.        
        Y : int array
            Label array where rows correspond to labels and columns
            correspond to samples. Entries in this matrix should only
            be +1 or -1.
        phi : dict
            Dictionary of outputs of binary-valued functions in an 
            ADT (i.e., value of samples at the ADT's output nodes).
            See parent function `Adaboost` for details on keys and 
            values.
        w : float array
            Array of weights over samples where rows correspond to
            labels and columns correspond to samples.

    Kwargs
        model : {**tree**,stump} 

    Returns
        path : int
            Index of the decision node to which the new feature
            should be connected in the ADT.
        feature : int
            Row index of data matrix `X` that corresponds to the
            selected feature.
        decision : {0,1}
            Output node of the decision node `path` to which the
            decision node corresponding to the new feature should be
            connected.
        threshold : int or float
            Threshold attribute of the decision node for the selected
            feature.

    .. note::
        * D kmers
        * N virus sequences
        * L Host Classes

    .. warning::
        * The code builds a list of all possible threshold values from the entire data matrix. This is a bad idea if the data  matrix has too many possible values.
        
        * The C code expects the arrays passed to be in C contiguous order. This needs to be generalized, using strides, since simple operations (like transposing) can change the array to Fortran contiguous. Click this `link <http://stackoverflow.com/   questions/4420622/how-to-account-for-column-contiguous-array-when-extending-numpy-with-c>`_ to see how to do this.

    """

    (D,N) = X.shape
    K = Y.shape[0]

    if model=='tree':
        keys = phi.keys()
        keys.sort()
        phi_array = np.array([output_node[0][0] for round in keys for output_node in phi[round]])
        order = [[key,key] for key in keys]
        order = [p for ps in order for p in ps]
        order.pop(0)
    elif model=='stump':
        keys = phi.keys()
        keys.sort()
        phi_array = np.array([output_node[0][0] for round in keys for output_node in phi[round]])

    # `Z` holds the loss for each decision rule `phi` being tested
    Z = np.zeros((phi_array.shape[0],D),dtype=float)
    thresholds = np.unique(X[:])
    results = np.zeros((3,),dtype='int')

    # parse the C code from get_new_function.c
    #f = open(os.cwd()+'get_new_function.c','r')
    f = open('get_new_function.c','r')
    C_code = '\n'.join([line for line in f if '//' not in line])
    f.close()

    support_code = "#include <math.h>"

    # the python code that calls the C code using weave.inline
    weave.inline(C_code, ['X','Y','phi_array','w','thresholds','results','Z'], \
        support_code=support_code, verbose=2, compiler='gcc')

    path = order[results[0]]
    feature = results[1]
    if results[0]:
        decision = 1-results[0]%2
    else:
        decision = 0
    threshold = results[2]

    return path, feature, decision, threshold
