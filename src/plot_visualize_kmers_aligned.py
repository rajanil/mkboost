import numpy as np
import cPickle
import matplotlib.pyplot as plot
from matplotlib.transforms import Bbox
from matplotlib.colors import colorConverter as convert
import pdb
import sys

def compile_hit_matrix(sequences, kmers, m):
    """this function compiles a matrix representation of where
    along a collection of sequence alignments, a list of kmers 
    are found, up to a given mismatch. the function also returns
    a matrix, where 1 denotes gaps in the alignment of the sequences.

    Arguments
        sequences : list
            List of tuples containing virus protein sequence alignements
            and virus host class
        kmers : list
            List of kmers selected by Adaboost
        m : int
            Allowed mismatch

    Returns
        hitmatrix : int array 
            `N_seq` x (`alignment_length`+1) x `N_kmer` array where `N_seq` is total 
            number of viruses, `alignment_length` is the distance along 
            the virus sequence alignment and `N_kmer` is the number of kmers.
        gapmatrix : int array
            `N_seq` x `alignment_length` array, where 1 in the array denotes a
            gap in the alignment.

    .. note::
        In generating this hit matrix, the virus sequences are
        aligned using COBALT, a web-based multiple alignment algorithm.

    """

    N_sequences = len(sequences)
    N_kmers = len(kmers)
    alignment_length = len(sequences[0][0])
    kmer_length = len(kmers[0])
    hit_matrix = np.zeros((N_sequences,alignment_length-kmer_length+2,N_kmers),dtype='float')
    gap_matrix = np.zeros((N_sequences,alignment_length-kmer_length+1,1),dtype='float')

    for index, seq in enumerate(sequences):
        # first column stores the virus class
        hit_matrix[index,0,:] = seq[1]
        alignment = seq[0]
        alignment_length = len(alignment)
        for c in range(alignment_length-kmer_length+1):
            if alignment[c]=="-":
                gap_matrix[index,c,0] = 1.
            for kidx, kmer in enumerate(kmers):
                sequence = alignment[c:].replace('-','')
                if len(sequence)>=kmer_length:
                    mismatch = (np.array(list(sequence[:kmer_length]))!=np.array(list(kmer))).sum()
                    if mismatch<=m:
                        left_col = c+1
                        right_col = left_col+3*kmer_length
                        hit_matrix[index,left_col:right_col,kidx] = 1.

    return hit_matrix, gap_matrix

def plot_hit_matrix(hit_matrix, gap_matrix, k, m, kmers):
    """this function visualizes the kmers along protein sequences,
    represented as a matrix, using `imshow`.

    Arguments
        hit_matrix : int array
            `N_seq` x `col_size` x `N_kmer` array where `N_seq` is total 
            number of viruses, `col_size` is the resolution along 
            the virus sequence and `N_kmer` is the number of kmers.
        gapmatrix : int array
            `N_seq` x `alignment_length` array, where 1 in the array denotes a
            gap in the alignment.
        k : int
            size of k-mers
        m : int
            allowed mismatch
        kmers : list
            k-mers selected by Adaboost

    Returns
        figure : matplotlib figure object

    """
        
    background_colors = {
        'white' :   np.array([255,255,255]).reshape(1,1,3)/255.,
        'black' :   np.array([0,0,0]).reshape(1,1,3)/255.,
         'grey' :   np.array([38,38,38]).reshape(1,1,3)/255.,
     'darkgrey' :   np.array([18,18,18]).reshape(1,1,3)/255.,
     'offwhite' :   np.array([225,225,225]).reshape(1,1,3)/255.,
    }
    text_color = 'k'
    bg_color = 'w'
    axis_label_fontsize = 7
    axis_tick_fontsize = 6
    title_fontsize = 8
    legend_fontsize = 4
    kmer_colors = ['red','green','blue','purple','magenta','orange','cyan','black','hotpink']

    class_labels = np.unique(hit_matrix[:,0,0]).astype(int)
    num_classes = class_labels.size
    (num_proteins,num_cols,ig) = hit_matrix.shape
    num_cols = num_cols-1

    # show a max of 9 kmers on the visualization
    V = min([9,len(kmers)])
    data = np.zeros((num_proteins,num_cols,3),dtype=float)

    # for each k-mer, generate a matrix where occurence
    # of the k-mer is indicated by the rgb of the color
    # assigned to that k-mer
    data += gap_matrix[:,:]*background_colors['offwhite']
    for i in range(V):
        data += hit_matrix[:,1:,i:i+1] * np.array(list(convert.to_rgb(kmer_colors[i]))) 

    # at every location a kmer is not found,
    # assign rgb of background color
    idx = hit_matrix.shape[0] 
    data = data + (1-(data.sum(2)>0)).reshape(idx,num_cols,1) * background_colors['white']

    # set figure size and resolution
    DPI = 500
    fig_resolution = (1706, 1280)
    fig_size = tuple([res/float(DPI) for res in fig_resolution])
    figure = plot.figure(figsize=fig_size, facecolor=bg_color, edgecolor=bg_color)
    subplot = figure.add_subplot(111)
    subplot.set_position([0.05,0.05,0.78,0.85])
    subplot.imshow(data,aspect='auto',interpolation='nearest')

    # set X-axis and Y-axis labels, tick sizes and tick labels
    for label in class_labels[:-1]:
        y_coord = (hit_matrix[:,0,0]==label).nonzero()[0].max() + 0.5
        subplot.plot([0,hit_matrix.shape[1]-1], [y_coord, y_coord], '-', color='gray', linewidth=0.1)

    subplot.axis([0, hit_matrix.shape[1]-1, 0, hit_matrix.shape[0]-1])
    subplot.set_xticks([0, hit_matrix.shape[1]/2, hit_matrix.shape[1]-1])
    subplot.set_xticklabels(('0','Location along alignment','12340'), \
        color=text_color, verticalalignment='center', fontsize=axis_tick_fontsize)
    for line in subplot.get_xticklines():
        line.set_markersize(0)
    y_labels = ('Invertebrate','Plant','Vertebrate')
    y_label_loc = []
    for c in class_labels:
        y_label_loc.append(int(np.mean((hit_matrix[:,0,0]==c).nonzero()[0])))
    subplot.set_yticks(y_label_loc)
    subplot.set_yticklabels(y_labels, rotation=90, color=text_color, \
        horizontalalignment='center', fontsize=axis_tick_fontsize)
    for line in subplot.get_yticklines():
        line.set_markersize(0)

    # set figure title
    figure.suptitle('k = %d, m = %d' % (k,m), x=0.95, y=0.95, color=text_color, \
        fontsize=title_fontsize, verticalalignment='center', horizontalalignment='right')

    # set a figtext bbox (outside the visualization subfigure) for legend
    kmer_locs = np.linspace(0.5+V/2*0.04,0.5-V/2*0.04,V)
    for kidx in range(V):
        kmer = kmers[kidx]
        plot.figtext(0.98, kmer_locs[kidx], kmer, fontsize=legend_fontsize, \
            color=kmer_colors[kidx], horizontalalignment='right', verticalalignment='center')

    return figure

if __name__=="__main__":

    # set project path and parameters
    project_path = '/proj/ar2384/picorna/'
    virus_family = 'picorna'
    data_path = '%s/cache/%s_protein/' % (project_path, virus_family)
    
    # values of k, m, T and fold are read from std input
    # T = max number of boosting rounds that will be parsed
    # fold = CV fold
    (k, m, T, fold) = map(int,sys.argv[1:5])
    
    # load kmers selected by boosting
    # if a kmer is selected multiple times (with different thresholds), 
    # it is only added once to the list.
    kmers = []
    f = open(data_path + virus_family + '_adt_%s_%d_%d_%d.pkl' % (virus_family,k,m,fold),'r')
    adt = cPickle.load(f)
    f.close()
    [kmers.append(adt[t][0].name) for t in range(T) if adt[t][0].name not in kmers] 

    # load virus sequence alignments
    p = open('%s/data/%s_sequence_alignment.fasta' %(project_path, virus_family),'r')
    sequences = []
    sequence = 'A'
    label = 0
    viruses = []
    
    for line in p:
        # indicates start of a new virus
        if '>' in line:
            # add previous protein to list of sequences
            sequences.append([sequence,label])
            row = line.strip().split()[1].split(':')
            virus_id = int(row[0])
            viruses.append(virus_id)
            label = int(row[1])
            sequence = ''
        else:
            sequence += line.strip()
    p.close()
    # pop out the first dummy sequence
    sequences.pop(0)

    # matrix representation of position along an alignment
    # where a specific k-mer is found, up to m mismatches
    # also output are the gaps in the alignment
    hit_matrix, gap_matrix = compile_hit_matrix(sequences,kmers,m)

    # save compiled data
    f = open('%s/%s_hitmatrix_aligned_%d_%d_%d.pkl' % (data_path, virus_family, k, m, fold),'w')
    cPickle.Pickler(f,protocol=2).dump(hit_matrix)
    cPickle.Pickler(f,protocol=2).dump(gap_matrix)
    cPickle.Pickler(f,protocol=2).dump(viruses)
    f.close()

    # group viruses with similar hosts together
    sort_indices = hit_matrix[:,0,0].argsort()

    # plot and save the visualization
    figure = plot_hit_matrix(hit_matrix[sort_indices,:,:], gap_matrix[sort_indices,:,:], k, m, kmers)
    fname = '%s/fig/%s_%s_kmer_visualization_aligned_%d_%d_%d.eps' % (project_path, virus_family, sequence_type, k, m, fold)
    figure.savefig(fname,dpi=(500),format='eps')
